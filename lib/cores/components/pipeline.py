from lib.errors import (
    PimInstructionUnsupportedError,
    PimInstructionMalformedError,
    PipelineExitCallbackNotDefinedError,
)
from lib.cores.instructions import Instruction, OpType
from lib.monad import DataWrapper, Ptr
from lib.cores.components.base import BaseCore
from typing import Any, Callable


class Stage:
    def __init__(
        self,
        children: list[Ptr[Stage]] | None,
        parent: Ptr[Stage] | None = None,
        name: str = "unnamed",
        tick_rule: Callable[[], bool] | None = None,
        propagate_rule: Callable[[Instruction], bool] | None = None,
        entry_side_effect: Callable[[Instruction], bool] | None = None,
        exit_side_effect: Callable[[Instruction], None] | None = None,
    ):
        # always initialize as empty
        self.ins: Instruction | None = None
        self.parent: Ptr[Stage] | None = parent
        self.children: list[Ptr[Stage]] | None = children
        if self.children is not None:
            for p in self.children:
                p().parent = Ptr(self)
        self.name: str = name
        self.tick_rule: Callable[[], bool] = (
            tick_rule if tick_rule is not None else lambda: False
        )

        # default behavior is to just return true so that we don't unnecessarily stall
        self.entry_side_effect: Callable[[Instruction], bool] = (
            entry_side_effect if entry_side_effect is not None else lambda _: True
        )
        self._entry_side_effect_done: bool = False

        self.exit_side_effect: Callable[[Instruction], None] = (
            exit_side_effect if exit_side_effect is not None else lambda _: None
        )

        empty_rule: Callable[[Instruction], bool] = lambda _: self.ins is None
        entry_done: Callable[[Instruction], bool] = lambda ins: (
            True if self._entry_side_effect_done else self.entry_side_effect(ins)
        )
        self.propagate_rule: Callable[[Instruction], bool] = (
            (lambda ins: (empty_rule(ins) and propagate_rule(ins) and entry_done(ins)))
            if propagate_rule is not None
            else empty_rule
        )

    def set_parent(self, p: Ptr[Stage]):
        self.parent = p

    def set_propagate_rule(self, rule: Callable[[Instruction], bool]):
        self.propagate_rule = lambda ins: (self.ins is None) and (rule(ins))

    def propagate(self):
        if not self._entry_side_effect_done and not self.ins is None:
            self._entry_side_effect_done = self.entry_side_effect(self.ins)
        if self.parent is not None:
            stage = self.parent()
            if stage.ins is not None and self.propagate_rule(stage.ins):
                ins = stage.pop()
                if ins is not None:
                    self.ins = ins
                    stage.exit_side_effect(self.ins)
                    self._entry_side_effect_done = False
                    self._entry_side_effect_done = self.entry_side_effect(self.ins)

    def pop(self) -> Instruction | None:
        ins = self.ins
        self.ins = None
        return ins

    def __str__(self) -> str:
        return self.name + " at " + str(self.ins)

    def is_empty(self) -> bool:
        return self.ins is None

    def tick(self):
        if self.tick_rule() and self.ins is not None:
            self.ins.tick()


class Pipeline:
    """
    This class is a flexible pipeline class, but it does have
    some constraints on the names of stages.

    First, every stage name must start with "st_". Second,
    any stage that has any level of data dependency is considered
    an 'execution stage' and therefore its name MUST start with
    "st_e_".

    As of the writing of this docstring, the last stage is
    the only stage at which instructions will be tested for being
    "done." While this does not match physical behavior, it
    does functionally match the expected behavior.
    """

    def __init__(
        self,
        core: BaseCore,
        stages: list[Stage],
        pipe_exit_cb: Callable[[Instruction], None] | None = None,
        # transitions: list[Callable[[Instruction, Pipeline], bool]]
    ):
        self.timestamp: int = 0
        self.stages: list[Stage] = stages
        self.stage_names: list[str] = [str(s) for s in stages]
        # self.transitions: list[Callable[[Instruction, Pipeline], bool]] = transitions
        self.finished_buffer: list[Instruction] = []
        self.leaf_nodes: list[Stage] = [s for s in stages if s.children is None]
        if pipe_exit_cb is None:

            def cb(_: Instruction) -> None:
                raise PipelineExitCallbackNotDefinedError("")

            self.pipe_exit_cb: Callable[[Instruction], None] = cb
        else:
            self.pipe_exit_cb: Callable[[Instruction], None] = pipe_exit_cb

    def set_pipeline_exit_callback(self, cb: Callable[[Instruction], None]):
        self.pipe_exit_cb = cb

    def tick(self):
        self.timestamp += 1
        # Drain leaf nodes
        for st in self.leaf_nodes:
            if st.ins is not None and st.ins.is_done():
                self.pipe_exit_cb(st.ins)
                st.ins = None

        # Propagate instructions along the pipeline
        for i in range(len(self.stages) - 1, -1, -1):
            self.stages[i].propagate()

        # Tick in reverse order to implicitly assume
        # a transparent register file latch
        for s in reversed(self.stages):
            s.tick()

    def try_load(self, ins: Instruction) -> bool:
        if self.stages[0].is_empty():
            ins.timestamp = self.timestamp
            self.stages[0].ins = ins
            return True
        return False

    def __str__(self) -> str:
        str_rep = ""
        for i, stage in enumerate(self.stages):
            if i > 0:
                str_rep += "\t"
            str_rep += f"{str(stage)} -> \n"

        return str_rep[:-5]

    def is_empty(self):
        return all(s.is_empty() for s in self.stages)


def mkDefaultStages(core: BaseCore) -> list[Stage]:
    def writeback_prop(ins: Instruction):
        return ins.is_done()

    writeback = Stage(children=None, name="writeback", propagate_rule=writeback_prop)

    def exe_prop(ins: Instruction) -> bool:
        return ins.is_warm() or ins.is_mem()

    def exe_entry(ins: Instruction) -> bool:
        if ins.is_mem() and not ins.is_warm():
            ins.start()
        return True

    execute = Stage(
        children=[Ptr(writeback)],
        name="execute",
        propagate_rule=exe_prop,
        entry_side_effect=exe_entry,
    )

    def read_prop(ins: Instruction) -> bool:
        stages = [execute, writeback]
        for s in stages:
            if s.ins is not None:
                for input in [ins.in_reg1, ins.in_reg2]:
                    if input != "" and input == s.ins.dst:
                        return False
                if ("gdl" in ins.list_operands() or ins.is_mem()) and (
                    s.ins.is_mem() or ("gdl" in s.ins.list_operands())
                ):
                    return False
        return True

    def read_exit(ins: Instruction):
        if not ins.is_mem():
            for op in [ins.in_reg1, ins.in_reg2]:
                # for the NOP case
                if op != "":
                    ins.set_state_by_operand_name(op, core.get_reg(op))

    def read_entry(ins: Instruction):
        if not ins.is_warm() and not ins.is_mem():
            ins.start()
        return True

    read = Stage(
        children=[Ptr(execute)],
        name="read",
        propagate_rule=read_prop,
        entry_side_effect=read_entry,
        exit_side_effect=read_exit,
    )

    def exe_tickrule():
        if execute.ins is None:
            return False
        return True

    execute.tick_rule = exe_tickrule

    decode = Stage(children=[Ptr(read)], name="decode")
    fetch = Stage(children=[Ptr(decode)], name="fetch")

    stages = [fetch, decode, read, execute, writeback]
    return stages


def mkEnhancedStages(core: BaseCore) -> list[Stage]:
    def writeback_prop(ins: Instruction):
        return ins.is_done()

    writeback = Stage(children=None, name="writeback", propagate_rule=writeback_prop)

    def mem_prop(ins: Instruction) -> bool:
        if ins.operation == OpType.READ or ins.operation == OpType.WRITE:
            return True
        return False

    read_prop: Callable[[Instruction], bool] = lambda ins: not mem_prop(ins)

    def mem_entry(ins: Instruction) -> bool:
        if (not ins.is_warm()) and (
            execute.ins is None or execute.ins.timestamp > ins.timestamp
        ):
            ins.start()
            return True
        return False

    mem = Stage(
        children=None, name="mem", propagate_rule=mem_prop, entry_side_effect=mem_entry
    )

    def exe_prop(ins: Instruction) -> bool:
        if writeback.ins is not None:
            if ins.in_reg1 == writeback.ins.dst or ins.in_reg2 == writeback.ins.dst:
                return False
        if mem.ins is not None:
            if mem.ins.timestamp < ins.timestamp:
                return False
        return ins.is_warm()

    execute = Stage(children=[Ptr(writeback)], name="execute", propagate_rule=exe_prop)

    def read_entry(ins: Instruction):
        if not (mem.ins is None) and mem.ins.timestamp < ins.timestamp:
            return False
        if not ins.is_warm():
            ins.start()
            for op in [ins.in_reg1, ins.in_reg2]:
                ins.set_state_by_operand_name(op, core.get_reg(op))
        return True

    read = Stage(
        children=[Ptr(execute)],
        name="read",
        propagate_rule=read_prop,
        entry_side_effect=read_entry,
    )

    def exe_tickrule():
        if execute.ins is None:
            return False
        if mem.ins is not None:
            if execute.ins.timestamp > mem.ins.timestamp:
                # print(execute.ins.operands)
                # if any(
                #     e_o == m_o for e_o, m_o in zip(execute.ins.operands, mem.ins.operands)
                # ):
                return False
        return True

    execute.tick_rule = exe_tickrule

    decode = Stage(children=[Ptr(read), Ptr(mem)], name="decode")
    fetch = Stage(children=[Ptr(decode)], name="fetch")

    stages = [fetch, decode, read, execute, mem, writeback]
    return stages
