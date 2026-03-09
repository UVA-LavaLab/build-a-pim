from lib.memsys import MemSystem
from collections import deque
from lib.cores.instructions import Instruction, OpType
from lib.monad import DataWrapper
from typing import Any


class Core:
    def __init__(self, location: tuple[int, int, int, int]):
        self.channel: int = location[0]
        self.rank: int = location[1]
        self.bankgroup: int = location[2]
        self.bank: int = location[3]

        self.gdl: DataWrapper = DataWrapper([], None)
        self.instruction_queue: deque[Instruction] = deque()
        self.cycle: int = 0

    def add_instruction(self, op: OpType, operands: list[int] | None = None):
        self.instruction_queue.append(Instruction(op, operands))

    def local_access(self, mem: MemSystem, addr: int):
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
                self.local_access(mem, instr.operands[0])
            case _:
                pass

    def ifail(self, cond: bool, errmsg: str):
        if cond:
            raise Exception(errmsg)
