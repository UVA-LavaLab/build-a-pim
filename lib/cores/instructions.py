from enum import Enum
from lib.monad import DataWrapper, DataStatus
from collections.abc import Callable
from typing import Any


class OpType(Enum):
    NOP = 0
    READ = 1
    WRITE = 2
    ADD = 3
    SUB = 4
    MUL = 5
    DIV = 6


class IState(Enum):
    COLD = 0
    WARM = 1
    DONE = 2


class Instruction:
    def __init__(self, op: OpType, operands: list[int] | None = None, deadline: int | None = None) -> None:
        self.operation: OpType = op
        self.operands: list[int] = operands if operands is not None else []
        self.state: IState = IState.COLD
        self.deadline: int = deadline if deadline is not None else 0

    def __str__(self):
        return str(self.operation) + " on " + str(self.operands) + " is " + str(self.state)

    def start(self):
        if self.state == IState.COLD:
            self.state = IState.WARM
        else:
            raise Exception("Instruction already started, cannot start again.")

    def finish(self):
        if self.state == IState.WARM:
            self.state = IState.DONE
        else:
            raise Exception(
                "Instruction cannot be finished, current state:", self.state
            )

    def is_ready(self, timestamp: int):
        return timestamp >= self.deadline
