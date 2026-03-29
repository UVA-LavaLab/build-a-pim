from lib.errors import PimCmdNotSupportedError
from lib.memsys import MemSystem
from collections import deque
from lib.cores.instructions import Instruction, OpType
from lib.monad import DataWrapper, DataSetter, Ptr
from lib.controller.commands import CommandType, Command
from typing import Any


class Pipeline:
    def __init__(self):
        self.active_instructions: deque[Instruction] = deque()
        self.p_mem: Ptr[MemSystem] = p_mem

    def try_fetch(self, ins: Instruction) -> bool:
        # TODO: fill this in (try to put an instruction into the pipeline)
        # if the current instruction cannot make progress, we need to not do that
        return False


class Core:
    supported_cmds: list[CommandType] = [
        CommandType.PIM_ADD,
        CommandType.PIM_SUB,
        CommandType.PIM_DIV,
        CommandType.PIM_MUL,
        CommandType.PIM_ABS,
    ]

    def __init__(
        self,
        location: tuple[int, int, int, int],
        p_mem: Ptr[MemSystem],
        scratchpad_access_time: int = 2,
    ):
        self.channel: int = location[0]
        self.rank: int = location[1]
        self.bankgroup: int = location[2]
        self.bank: int = location[3]
        self.p_mem: Ptr[MemSystem] = p_mem

        self.gdl: DataWrapper = DataWrapper([], None)
        self.next_gdl: DataWrapper = DataWrapper([], None)
        self.instruction_queue: deque[Instruction] = deque()
        self.cycle: int = -1
        self.spad_acc_time: int = scratchpad_access_time
        self.active_instructions: list[Instruction] = []

    def add_instruction(self, op: OpType, operands: list[int] | None = None):
        self.instruction_queue.append(Instruction(op, operands))

    def local_mem_op(self, addr: int, is_write: bool) -> DataSetter | None:
        if is_write:
            ds = DataSetter(self.gdl)
            self.p_mem()[self.channel, self.rank, self.bankgroup, self.bank, addr] = ds
            return ds
        else:
            self.next_gdl = self.p_mem()[
                self.channel, self.rank, self.bankgroup, self.bank, addr
            ]
            return None

    def update_data_states(self):
        _ = self.gdl.is_ready()
        _ = self.next_gdl.is_ready()

    def can_add_to_active(self, instr: Instruction) -> bool:
        return not instr.operation in [i.operation for i in self.active_instructions]

    def parse_cmd(self, cmd: Command):
        return

    def tick(self, cmd: Command | None = None):
        if cmd is not None:
            if cmd.cmdtype not in self.supported_cmds:
                raise PimCmdNotSupportedError(
                    f"{self.__class__.__name__} does not support command type {cmd.cmdtype}."
                )
            self.parse_cmd(cmd)
        print("active:", [str(i) for i in self.active_instructions])
        active_instr: list[Instruction] = []
        # TODO: enhance performance here
        for instr in self.active_instructions:
            print(instr.operation)
            instr.tick()
            print(instr.is_done())
            if instr.is_done():
                self.call_end_handler(instr)
                # implicitly removes the instruction
                # from the active instruction list
                continue
            active_instr.append(instr)
        self.active_instructions = active_instr
        # TODO: fix this to ensure no data dependencies are violated
        if len(self.instruction_queue) > 0 and self.can_add_to_active(
            self.instruction_queue[0]
        ):
            self.call_start_handler(self.instruction_queue.popleft())
        self.cycle += 1
        print("new active:", [str(i) for i in self.active_instructions])

    def call_start_handler(self, instr: Instruction):
        self.active_instructions.append(instr)
        match instr.operation:
            case OpType.READ:
                self.ifail(
                    len(instr.operands) < 1,
                    "No argument supplied for instruction READ.",
                )

                def idcb():
                    return self.next_gdl.is_ready()

                self.active_instructions[-1].is_done = idcb
                _ = self.local_mem_op(instr.operands[0], False)
            case OpType.WRITE:
                self.ifail(
                    len(instr.operands) < 1,
                    "No argument supplied for instruction WRITE.",
                )
                self.write_queue.append(self.local_mem_op(instr.operands[0], True))
            case _:
                pass

    def call_end_handler(self, instr: Instruction):
        if self.next_gdl.is_ready:
            self.gdl = self.next_gdl
        match instr.operation:
            case _:
                pass

    def ifail(self, cond: bool, errmsg: str):
        if cond:
            raise Exception(errmsg)
