from enum import Enum
from lib.monad import DataWrapper, DataStatus
from collections.abc import Callable
from typing import Any


class OpType(Enum):
    # Mem and control flow instructions
    NOP = 0
    READ = 1
    WRITE = 2
    JUMP = 3
    JNZ = 4
    JNE = 5
    # Vector instructions (f(vec a', vec b') -> vec c')
    VEC_ADD = 6
    VEC_SUB = 7
    VEC_MUL = 8
    VEC_DIV = 9
    VEC_ABS = 10
    VEC_NOT = 11
    VEC_AND = 12
    VEC_OR = 13
    VEC_XOR = 14
    VEC_XNOR = 15
    VEC_MIN = 16
    VEC_MAX = 17
    # Scalar instructions (f(a', vec b') -> vec c') or (f(vec a', b') -> vec c')
    SCALAR_ADD = 18
    SCALAR_SUB = 19
    SCALAR_MUL = 20
    SCALAR_DIV = 21
    SCALAR_AND = 22
    SCALAR_OR = 23
    SCALAR_XOR = 24
    SCALAR_XNOR = 25
    SCALAR_MIN = 26
    SCALAR_MAX = 27
    # Relational operations (f(vec a', vec b') -> vec bool)
    REL_GT = 28
    REL_LT = 29
    REL_EQ = 30
    REL_NE = 31
    SCALAR_REL_GT = 32
    SCALAR_REL_LT = 33
    SCALAR_REL_EQ = 34
    SCALAR_REL_NE = 35
    # Reduction operations (f(vec a', vec b') -> c')
    RED_ADD = 36
    RED_MAX = 37
    RED_MIN = 38
    # Misc. Instructions
    VEC_POPCOUNT = 39
    VEC_MAC = 40
    # TODO: provide more instructions here


class IState(Enum):
    COLD = 0
    WARM = 1
    DONE = 2


class Instruction:
    def __init__(
        self,
        op: OpType,
        timestamp: int = 0,
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
        self.timestamp: int = 0

        def idcb() -> bool:
            if is_done_cb is not None:
                rval = is_done_cb()
            else:
                rval = self.completion_time <= 0
            self.state = IState.DONE if rval else self.state
            return rval

        self.is_done: Callable[[], bool] = idcb
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
        self.op_vals: dict[str | int, DataWrapper] = {}

    def fetch_operands(self, ind: int) -> DataWrapper:
        return self.op_vals[self.operands[ind]]

    def tick(self):
        self.completion_time -= 1

    def __str__(self):
        return (
            str(self.operation) + " on " + str(self.operands) + " is " + str(self.state) + " with timestamp " + str(self.timestamp) + f" ct: {self.completion_time}"
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
