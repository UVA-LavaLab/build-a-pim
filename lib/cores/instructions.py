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
    def __init__(
        self,
        op: OpType,
        operands: list[int | str] | None = None,
        completion_time: int | None = None,
        is_done_cb: None | Callable[[], bool] = None,
        ret: None | Callable[[], DataWrapper] = None,
    ) -> None:
        self.operation: OpType = op
        self.operands: list[int | str] = operands if operands is not None else []
        self.completion_time: int = (
            completion_time if completion_time is not None else 0
        )

        self.is_done: Callable[[], bool] = (
            is_done_cb if is_done_cb is not None else lambda: self.completion_time <= 0
        )
        self.data: DataWrapper = DataWrapper([])

        if ret is not None:
            self.ret: Callable[[], DataWrapper] = ret
        else:

            def noret():
                return self.data

            self.ret = noret

        # FIXME: considering removing this
        self.state: IState = IState.COLD
        self.start_cb: Callable[[], None] = lambda: None

    def tick(self):
        self.completion_time -= 1

    def __str__(self):
        return (
            str(self.operation) + " on " + str(self.operands) + " is " + str(self.state)
        )

    def is_mem(self):
        return self.operation == OpType.READ or self.operation == OpType.WRITE

    def start(self):
        if self.state == IState.COLD:
            self.start_cb()
            self.state = IState.WARM
        else:
            raise Exception("Instruction already started, cannot start again.")

    def is_warm(self):
        return not self.state == IState.COLD

    # FIXME: considering removing this
    def finish(self):
        if self.state == IState.WARM:
            self.state = IState.DONE
        else:
            raise Exception(
                "Instruction cannot be finished, current state:", self.state
            )
