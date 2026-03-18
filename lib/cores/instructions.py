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
    def __init__(self, op: OpType, operands: list[int] | None = None, deadline: int | None = None, is_done_cb: None | Callable[[], bool]=None) -> None:
        self.operation: OpType = op
        self.operands: list[int] = operands if operands is not None else []
        self.deadline: int = deadline if deadline is not None else 0
        def idcb() -> bool:
            return self.deadline <= 0
        self.is_done: Callable[[], bool] = is_done_cb if is_done_cb is not None else idcb

        # FIXME: considering removing this
        self.state: IState = IState.COLD

    def tick(self):
        self.deadline -= 1

    def __str__(self):
        return str(self.operation) + " on " + str(self.operands) + " is " + str(self.state)

    # FIXME: considering removing this
    def start(self):
        if self.state == IState.COLD:
            self.state = IState.WARM
        else:
            raise Exception("Instruction already started, cannot start again.")

    # FIXME: considering removing this
    def finish(self):
        if self.state == IState.WARM:
            self.state = IState.DONE
        else:
            raise Exception(
                "Instruction cannot be finished, current state:", self.state
            )
