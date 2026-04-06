from lib.errors import (
    PimCmdNotSupportedError,
    PimInstructionUnsupportedError,
    PimInstructionMalformedError,
)
from lib.memsys import MemSystem
from collections import deque
from lib.cores.instructions import Instruction, OpType
from lib.monad import DataWrapper, DataSetter, Ptr
from lib.controller.commands import CommandType, Command
from typing import Any, Callable
from functools import reduce


class Stage:
    def __init__(
        self,
        children: list[Ptr[Stage]] | None,
        parent: Ptr[Stage] | None = None,
        name: str = "unnamed",
        tick_rule: Callable[[], bool] | None = None,
        propagate_rule: Callable[[Instruction], bool] | None = None,
        entry_side_effect: Callable[[Instruction], bool] | None = None,
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
        self._entry_side_effect_done = False

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


def mkDefaultStages(core: Core) -> list[Stage]:
    def writeback_prop(ins: Instruction):
        return ins.is_done()

    writeback = Stage(children=None, name="writeback", propagate_rule=writeback_prop)

    def exe_prop(ins: Instruction) -> bool:
        return ins.is_warm() or ins.is_mem()

    def exe_entry(ins: Instruction) -> bool:
        if ins.is_mem() and not ins.is_warm():
            ins.start()
        return True

    execute = Stage(children=[Ptr(writeback)], name="execute", propagate_rule=exe_prop, entry_side_effect=exe_entry)

    def read_prop(ins: Instruction) -> bool:
        stages = [writeback, execute]
        for s in stages:
            if s.ins is not None:
                for operand in ins.operands:
                    if operand in s.ins.operands:
                        return False
        return True

    def read_entry(ins: Instruction):
        if not ins.is_warm() and not ins.is_mem():
            ins.start()
            for operand in ins.operands:
                if isinstance(operand, str):
                    ins.op_vals[operand] = core.get_reg(operand)
                else:
                    ins.op_vals[operand] = core.gdl
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
        return True

    execute.tick_rule = exe_tickrule

    decode = Stage(children=[Ptr(read)], name="decode")
    fetch = Stage(children=[Ptr(decode)], name="fetch")

    stages = [fetch, decode, read, execute, writeback]
    return stages

def mkEnhancedStages(core: Core) -> list[Stage]:
    def writeback_prop(ins: Instruction):
        return ins.is_done()

    writeback = Stage(children=None, name="writeback", propagate_rule=writeback_prop)

    def exe_prop(ins: Instruction) -> bool:
        return ins.is_warm()

    execute = Stage(children=[Ptr(writeback)], name="execute", propagate_rule=exe_prop)

    def mem_prop(ins: Instruction) -> bool:
        if ins.operation == OpType.READ or ins.operation == OpType.WRITE:
            return True
        return False

    read_prop: Callable[[Instruction], bool] = lambda ins: not mem_prop(ins)

    def mem_entry(ins: Instruction) -> bool:
        if (not ins.is_warm()) and (execute.ins is None or execute.ins.timestamp > ins.timestamp):
            ins.start()
            return True
        return False

    mem = Stage(
        children=None, name="mem", propagate_rule=mem_prop, entry_side_effect=mem_entry
    )

    def read_entry(ins: Instruction):
        if not (mem.ins is None) and mem.ins.timestamp < ins.timestamp:
            return False
        if not ins.is_warm():
            ins.start()
            for operand in ins.operands:
                if isinstance(operand, str):
                    ins.op_vals[operand] = core.get_reg(operand)
                else:
                    ins.op_vals[operand] = core.gdl
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
        core: Core,
        stages: list[Stage],
        pipe_exit_cb: Callable[[Instruction], None] | None = None,
        # transitions: list[Callable[[Instruction, Pipeline], bool]]
    ):
        self.timestamp: int = 0
        self.stages: list[Stage] = stages
        self.stage_names: list[str] = [str(s) for s in stages]
        # self.transitions: list[Callable[[Instruction, Pipeline], bool]] = transitions
        self.finished_buffer: list[Instruction] = []
        if pipe_exit_cb is None:

            # TODO: determine how to do bounds checking for addresses passed here,
            # currently just assumes that the requested address is in the GDL already
            # TODO: add type checking before performing operations
            def eval(f, ins: Instruction):
                assert len(ins.operands) >= 2
                if isinstance(ins.operands[0], str) and isinstance(
                    ins.operands[1], int
                ):
                    dst = ins.op_vals[ins.operands[0]]
                    for i in range(len(ins.op_vals[ins.operands[1]].data)):
                        dst[i] = f(dst[i], ins.op_vals[ins.operands[1]][i])
                    core.set_reg("gdl", dst)
                elif isinstance(ins.operands[0], int) and isinstance(
                    ins.operands[1], str
                ):
                    reg = ins.op_vals[ins.operands[1]]
                    for i in range(len(ins.op_vals[ins.operands[0]].data)):
                        ins.op_vals[ins.operands[0]][i] = f(
                            ins.op_vals[ins.operands[0]][i], reg[i]
                        )
                    core.set_reg("gdl", reg)
                elif isinstance(ins.operands[0], str) and isinstance(
                    ins.operands[1], str
                ):
                    reg0 = ins.fetch_operands(0)
                    reg1 = ins.fetch_operands(1)
                    for i in range(len(reg0.data)):
                        reg0[i] = f(reg0[i], reg1[i])
                    core.set_reg(ins.operands[0], reg0)
                else:
                    raise PimInstructionUnsupportedError(
                        "Arithmetic operations between two memory locations unsupported."
                    )

            def pecb(ins: Instruction):
                match ins.operation:
                    case OpType.READ | OpType.WRITE:
                        core.gdl = ins.ret()
                        if len(ins.operands) > 1 and isinstance(ins.operands[1], str):
                            setattr(core, ins.operands[1], core.gdl)
                    case OpType.ADD:
                        eval(lambda x, y: x + y, ins)
                    case OpType.SUB:
                        eval(lambda x, y: x - y, ins)
                    case OpType.MUL:
                        eval(lambda x, y: x * y, ins)
                    case OpType.ACC:
                        if (
                            len(ins.operands) < 2
                            or not isinstance(ins.operands[0], str)
                            or not isinstance(ins.operands[0], str)
                        ):
                            raise PimInstructionMalformedError(
                                "Accumulating to a non-value register is currently unsupported."
                            )
                        else:
                            if isinstance(ins.operands[1], str):
                                vreg = getattr(core, ins.operands[1])
                                acc = 0
                                for i in range(len(vreg.data)):
                                    acc += vreg[i]
                                setattr(core, ins.operands[0], acc)
                    case OpType.DIV:
                        eval(lambda x, y: type(x)(x / y), ins)
                    case _:
                        pass

            self.pipe_exit_cb: Callable[[Instruction], None] = pecb
        else:
            self.pipe_exit_cb: Callable[[Instruction], None] = pipe_exit_cb

    # def check_data_dependency(self, ins: Instruction, pos: int) -> bool:
    #     """
    #     Returns true when there IS a data dependency further down the pipeline.
    #     """
    #     for i in [getattr(self, s) for s in self.stages[pos : len(self.stages)]]:
    #         if i is not None:
    #             for operand in i.operands:
    #                 if operand in ins.operands:
    #                     return True

    # return False

    # def check_unique_mem_op(self, pos: int) -> bool:
    #     """
    #     Returns true when there IS another memory operation ahead of the passed position.
    #     """
    #     for i in [getattr(self, s) for s in self.stages[pos : len(self.stages)]]:
    #         if i is not None:
    #             if i.operation == OpType.READ or i.operation == OpType.WRITE:
    #                 return True
    #
    #     return False

    def tick(self):
        self.timestamp += 1
        st_wb: Stage = self.stages[-1]
        st_mem: Stage = self.stages[-2]
        if st_mem.ins is not None and st_mem.ins.is_done():
            self.pipe_exit_cb(st_mem.ins)
            st_mem.ins = None
        if st_wb.ins is not None and st_wb.ins.is_done():
            self.pipe_exit_cb(st_wb.ins)
            st_wb.ins = None
        for i in range(len(self.stages) - 1, -1, -1):
            self.stages[i].propagate()

        for s in self.stages:
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


class Core:
    supported_cmds: list[CommandType] = [
        CommandType.PIM_ADD,
        CommandType.PIM_SUB,
        CommandType.PIM_DIV,
        CommandType.PIM_MUL,
        CommandType.PIM_ABS,
    ]
    isa: list[OpType] = [
        OpType.NOP,
        OpType.ADD,
        OpType.SUB,
        OpType.MUL,
        OpType.DIV,
        OpType.ACC,
        OpType.READ,
        OpType.WRITE,
    ]
    timings: dict[OpType, int] = {
        OpType.NOP: 1,
        OpType.ADD: 1,
        OpType.SUB: 1,
        OpType.MUL: 2,
        OpType.DIV: 2,
        OpType.ACC: 1,
        # these timings do not matter since we handle them externally
        OpType.READ: 0,
        OpType.WRITE: 0,
    }

    def __init__(
        self,
        location: tuple[int, int, int, int],
        p_mem: Ptr[MemSystem],
        scratchpad_access_time: int = 2,
        registers: list[str] | None = None,
        vec_registers: list[str] | None = None,
        pipeline_stages: list[Stage] | None = None,
    ):
        self.channel: int = location[0]
        self.rank: int = location[1]
        self.bankgroup: int = location[2]
        self.bank: int = location[3]
        self.p_mem: Ptr[MemSystem] = p_mem

        self.gdl: DataWrapper = DataWrapper([], None)
        self.instruction_queue: deque[Instruction] = deque()
        self.cycle: int = -1
        self.spad_acc_time: int = scratchpad_access_time
        self.pipeline: Pipeline = Pipeline(
            self,
            (mkEnhancedStages(self) if pipeline_stages is None else pipeline_stages),
        )
        self.reg: dict[str, DataWrapper] = {}
        if registers is None:
            self.registers: list[str] = ["regA", "regB", "regC"]
        else:
            self.registers = registers

        for r in self.registers:
            setattr(self, r, 0)
            # TODO: determine how to relax this
            self.__class__.__annotations__[r] = int | float

        if vec_registers is None:
            self.vec_registers: list[str] = ["reg_vA", "reg_vB", "reg_vC"]
        else:
            self.vec_registers = vec_registers

        for r in self.vec_registers:
            setattr(self, r, DataWrapper([]))
            self.__class__.__annotations__[r] = DataWrapper | None

    def add_instruction(self, op: OpType, operands: list[int | str] | None = None):
        if operands is None:
            operands = []
        self.instruction_queue.append(
            Instruction(op, operands=operands, completion_time=self.timings[op])
        )

    def local_mem_op(self, addr: int, is_write: bool) -> DataWrapper | None:
        if is_write:
            return self.p_mem().set(
                (self.channel, self.rank, self.bankgroup, self.bank, addr), self.gdl
            )
        else:
            return None

    def update_data_states(self):
        _ = self.gdl.is_ready()

    def parse_cmd(self, cmd: Command) -> list[Instruction] | None:
        return None

    def get_reg(self, reg: str) -> DataWrapper:
        rval: DataWrapper | None = getattr(self, reg)
        if rval is None:
            rval = DataWrapper([])
        return rval

    def set_reg(self, reg: str, val: DataWrapper):
        setattr(self, reg, val)

    def tick(self, cmd: Command | None = None):
        self.pipeline.tick()
        if cmd is not None:
            if cmd.cmdtype not in self.supported_cmds:
                raise PimCmdNotSupportedError(
                    f"{self.__class__.__name__} does not support command type {cmd.cmdtype}."
                )
            # TODO: use the parsed cmd
            _ = self.parse_cmd(cmd)
        if len(self.instruction_queue) > 0 and self.pipeline.try_load(
            self.instruction_queue[0]
        ):
            self.call_start_handler(self.instruction_queue.popleft())
        self.cycle += 1

    def call_start_handler(self, instr: Instruction):
        match instr.operation:
            case OpType.READ:
                self.ifail(
                    len(instr.operands) < 1,
                    "No argument supplied for instruction READ.",
                )

                # safety check
                assert isinstance(instr.operands[0], int)

                def scb():
                    instr.data = self.p_mem().get(
                        (
                            self.channel,
                            self.rank,
                            self.bankgroup,
                            self.bank,
                            # makes the interpreter not freak out
                            int(instr.operands[0]),
                        )
                    )

                instr.start_cb = scb
                instr.is_done = lambda: instr.data.is_ready()

            case OpType.WRITE:
                self.ifail(
                    len(instr.operands) < 1,
                    "No argument supplied for instruction WRITE.",
                )

                assert isinstance(instr.operands[0], int)

                def scb():
                    instr.data = self.p_mem().set(
                        (
                            self.channel,
                            self.rank,
                            self.bankgroup,
                            self.bank,
                            int(instr.operands[0]),
                        ),
                        self.gdl,
                    )

                instr.start_cb = scb
                instr.is_done = lambda: instr.data.is_ready()
            case _:
                pass

    def ifail(self, cond: bool, errmsg: str):
        if cond:
            raise Exception(errmsg)
