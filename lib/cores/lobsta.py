from lib.errors import PimCmdNotSupportedError, PimInstructionUnsupportedError, PimInstructionMalformedError
from lib.memsys import MemSystem
from collections import deque
from lib.cores.instructions import Instruction, OpType
from lib.monad import DataWrapper, DataSetter, Ptr
from lib.controller.commands import CommandType, Command
from typing import Any, Callable
from functools import reduce


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
        stages: list[str],
        pipe_exit_cb: Callable[[Instruction], None] | None = None,
        # transitions: list[Callable[[Instruction, Pipeline], bool]]
    ):
        self.stages: list[str] = stages
        self.exe_stages: list[str] = [s for s in stages if s.startswith("st_e_")]
        # self.transitions: list[Callable[[Instruction, Pipeline], bool]] = transitions
        for s in stages:
            setattr(self, s, None)
            self.__class__.__annotations__[s] = Instruction | None
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
                    dst = getattr(core, ins.operands[0])
                    for i in range(len(core.gdl.data)):
                        dst[i] = f(dst[i], core.gdl[i])
                    setattr(core, ins.operands[0], dst)
                elif isinstance(ins.operands[0], int) and isinstance(
                    ins.operands[1], str
                ):
                    reg = getattr(core, ins.operands[1])
                    for i in range(len(core.gdl.data)):
                        core.gdl[i] = f(core.gdl[i], reg[i])
                elif isinstance(ins.operands[0], str) and isinstance(
                    ins.operands[1], str
                ):
                    reg0 = getattr(core, ins.operands[0])
                    reg1 = getattr(core, ins.operands[0])
                    for i in range(len(reg0.data)):
                        reg0[i] = f(reg0[i], reg1[i])
                    setattr(core, ins.operands[0], reg0)
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
                        if len(ins.operands) < 2 or not isinstance(ins.operands[0], str) or not isinstance(ins.operands[0], str):
                            raise PimInstructionMalformedError("Accumulating to a non-value register is currently unsupported.")
                        else:
                            if isinstance(ins.operands[1], str):
                                vreg = getattr(core, ins.operands[1])
                                acc = 0
                                for i in range(len(vreg.data)):
                                    acc += vreg[i]
                                setattr(core, ins.operands[0], acc)
                    # TODO: prevent automatic type coercion to float when needed
                    case OpType.DIV:
                        eval(lambda x, y: x / y, ins)
                    case _:
                        pass

            self.pipe_exit_cb: Callable[[Instruction], None] = pecb
        else:
            self.pipe_exit_cb: Callable[[Instruction], None] = pipe_exit_cb

    def check_data_dependency(self, ins: Instruction, pos: int) -> bool:
        """
        Returns true when there IS a data dependency further down the pipeline.
        """
        for i in [getattr(self, s) for s in self.stages[pos : len(self.stages)]]:
            if i is not None:
                for operand in i.operands:
                    if operand in ins.operands:
                        return True

        return False

    def check_unique_mem_op(self, pos: int) -> bool:
        """
        Returns true when there IS another memory operation ahead of the passed position.
        """
        for i in [getattr(self, s) for s in self.stages[pos : len(self.stages)]]:
            if i is not None:
                if i.operation == OpType.READ or i.operation == OpType.WRITE:
                    return True

        return False

    def tick(self):
        for e in self.exe_stages:
            cur_val: Instruction | None = getattr(self, e)
            if cur_val is not None:
                if not cur_val.is_warm():
                    cur_val.start()
                cur_val.tick()
        last_stage: Instruction | None = getattr(self, self.stages[-1])
        if isinstance(last_stage, Instruction) and last_stage.is_done():
            self.pipe_exit_cb(last_stage)
            # self.finished_buffer.append(last_stage)
            setattr(self, self.stages[-1], None)
        for i in range(len(self.stages) - 1, 0, -1):
            prev_stage_val: Instruction | None = getattr(self, self.stages[i - 1])
            if self.stages[i].startswith("st_e_") and prev_stage_val is not None:
                if self.check_data_dependency(prev_stage_val, i):
                    continue
                if prev_stage_val.is_mem() and self.check_unique_mem_op(i):
                    continue
                cur_val: Instruction = getattr(self, self.stages[i])
            if getattr(self, self.stages[i]) is None:
                setattr(self, self.stages[i], prev_stage_val)
                setattr(self, self.stages[i - 1], None)

    def try_load(self, ins: Instruction) -> bool:
        if getattr(self, self.stages[0]) is None:
            setattr(self, self.stages[0], ins)
            return True
        return False

    def __str__(self) -> str:
        str_rep = ""
        for stage_name in self.stages:
            str_rep += f"{stage_name}[{str(getattr(self, stage_name))}] -> "

        return str_rep[:-4]

    def is_empty(self):
        return reduce(
            lambda x, y: x and (y is None),
            [getattr(self, stage) for stage in self.stages],
            True,
        )


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
        self.pipeline: Pipeline = Pipeline(self, ["st_f", "st_e_exe", "st_e_mem"])
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
        self.instruction_queue.append(
            Instruction(op, operands, completion_time=self.timings[op])
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
                # _ = self.local_mem_op(instr.operands[0], False)
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
            case OpType.ADD:
                pass
            case _:
                pass

    def ifail(self, cond: bool, errmsg: str):
        if cond:
            raise Exception(errmsg)
