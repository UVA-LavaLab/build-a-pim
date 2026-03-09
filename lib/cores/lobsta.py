from lib.memsys import MemSystem
from collections import deque
from lib.cores.instructions import Instruction, OpType
from lib.monad import DataWrapper
from typing import Any


class Core:
    def __init__(
        self, location: tuple[int, int, int, int], scratchpad_access_time: int=2
    ):
        self.channel: int = location[0]
        self.rank: int = location[1]
        self.bankgroup: int = location[2]
        self.bank: int = location[3]

        self.gdl: DataWrapper = DataWrapper([], None)
        self.instruction_queue: deque[Instruction] = deque()
        self.cycle: int = 0
        self.spad_acc_time: int = scratchpad_access_time

    def add_instruction(self, op: OpType, operands: list[int] | None = None):
        self.instruction_queue.append(Instruction(op, operands))

    def local_access(self, mem: MemSystem, addr: int, is_write: bool):
        if is_write:
            mem[self.channel, self.rank, self.bankgroup, self.bank, addr] = self.gdl
        else:
            self.gdl = mem[self.channel, self.rank, self.bankgroup, self.bank, addr]

    def tick(self, mem: MemSystem):
        if self.instruction_queue[0].deadline <= self.cycle:
            self.call_handler(mem, self.instruction_queue.popleft())
        self.cycle += 1

    def call_handler(self, mem: MemSystem, instr: Instruction):
        match instr.operation:
            case OpType.READ:
                self.ifail(
                    len(instr.operands) < 1,
                    "No argument supplied for instruction READ.",
                )
                self.local_access(mem, instr.operands[0], False)
            case OpType.WRITE:
                self.ifail(
                    len(instr.operands) < 1,
                    "No argument supplied for instruction WRITE.",
                )
                self.local_access(mem, instr.operands[0], True)
            case _:
                pass

    def ifail(self, cond: bool, errmsg: str):
        if cond:
            raise Exception(errmsg)
