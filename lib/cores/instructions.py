from enum import Enum
from lib.monad import DataWrapper
from collections.abc import Callable
from typing import override, Any
from lib.errors import PimInvalidRegisterIDError
import numpy as np
import numpy.typing as npt


class OpType(Enum):
    # Mem and control flow instructions
    NOP = 0
    READ = 1
    WRITE = 2
    JUMP = 3
    JNZ = 4
    JNE = 5
    CMP = 6
    # Vector instructions (f(vec a', vec b') -> vec c')
    VEC_ADD = 7
    VEC_SUB = 8
    VEC_MUL = 9
    VEC_DIV = 10
    VEC_ABS = 11
    VEC_NOT = 12
    VEC_AND = 13
    VEC_OR = 14
    VEC_XOR = 15
    VEC_XNOR = 16
    VEC_MIN = 17
    VEC_MAX = 18
    VEC_POPCOUNT = 19
    VEC_MAC = 20
    # Scalar instructions (f(a', vec b') -> vec c') or (f(vec a', b') -> vec c')
    SCALAR_ADD = 21
    SCALAR_SUB = 22
    SCALAR_MUL = 23
    SCALAR_DIV = 24
    SCALAR_AND = 25
    SCALAR_OR = 26
    SCALAR_XOR = 27
    SCALAR_XNOR = 28
    SCALAR_MIN = 29
    SCALAR_MAX = 30
    # Relational operations (f(vec a', vec b') -> vec bool)
    REL_GT = 31
    REL_LT = 32
    REL_EQ = 33
    REL_NE = 34
    # Scalar relational operations (f(vec a', vec b') -> vec bool)
    SCALAR_REL_GT = 35
    SCALAR_REL_LT = 36
    SCALAR_REL_EQ = 37
    SCALAR_REL_NE = 38
    # Reduction operations (f(vec a', vec b') -> c')
    RED_ADD = 39
    RED_MAX = 40
    RED_MIN = 41
    # Misc. Instructions
    GET_ACTIVE_ROW = 42
    TO_SWITCHING_MODE = 43
    TO_PAUSED_MODE = 44
    TO_PIM_MODE = 45
    MEM_READ = 46
    MEM_WRITE = 47
    SCRATCH_READ = 48
    SCRATCH_WRITE = 49
    # TODO: provide more instructions here


class IState(Enum):
    COLD = 0
    WARM = 1
    DONE = 2


# We acknowledge 4 types of instruction classes:
# - control flow instruction classes (which can be single- or multi-operand  (cmp x, y or jnz addr))
# - vector operations (which are multi-operand: vec_reg1, vec_reg2 -> vec_reg3)
# - scalar operations (which are multi-operand: reg, vec_reg1 -> vec_reg2)
# - reduction operations (which are single-operand: vec_reg -> reg)
# For memory operations, we need to be able to pass an address
# and possibly a reg (addr, Option[reg])
#
# To support non-SIMD operations (or different vector widths),
# we should optionally accept an offset with each register.
class Instruction:
    def __init__(
        self,
        op: OpType,
        timestamp: int = 0,
        in_reg1: str | None = None,
        in_reg2: str | None = None,
        dst: str | None = None,
        addr: int | None = None,
        start_index: int | None = None,
        completion_time: int | None = None,
        is_done_cb: None | Callable[[], bool] = None,
        ret: None | Callable[[], DataWrapper] = None,
        emit: bool = False,
        dtype: npt.DTypeLike = np.int32,
    ) -> None:
        self.operation: OpType = op
        self.in_reg1: str = "" if in_reg1 is None else in_reg1
        self.in_reg2: str = "" if in_reg2 is None else in_reg2
        self.dst: str = "" if dst is None else dst

        self.addr: int = -1 if addr is None else addr
        self.completion_time: int = (
            completion_time if completion_time is not None else 0
        )
        self.start_index: int | None = start_index
        self.timestamp: int = 0
        self.emit: bool = emit

        def idcb() -> bool:
            if is_done_cb is not None:
                rval = is_done_cb()
            else:
                rval = self.completion_time <= 0
            self.state = IState.DONE if rval else self.state
            return rval

        self.is_done: Callable[[], bool] = idcb
        self.data: DataWrapper = DataWrapper([])
        self.dtype: npt.DTypeLike = dtype

        if ret is not None:
            self.ret: Callable[[], DataWrapper] = ret
        else:

            def noret():
                return self.data

            self.ret = noret

        # FIXME: considering removing this
        self.state: IState = IState.COLD
        self.start_cb: Callable[[], None] = lambda: None
        self._op_vals: dict[str | int, DataWrapper] = {}

    def set_is_done(self, f: Callable[[], bool]):
        def idcb() -> bool:
            rval = f()
            self.state = IState.DONE if rval else self.state
            return rval
        self.is_done = idcb


    def set_state_by_operand_name(self, op: str, val: DataWrapper | Any) -> None:
        self._op_vals[op] = val

    def get_state_by_operand_id(self, ind: int) -> DataWrapper:
        match ind:
            case 0:
                operand = self.in_reg1
            case 1:
                operand = self.in_reg2
            case 2:
                operand = self.dst
            case _:
                raise PimInvalidRegisterIDError(
                    f"Register ID {ind} not supported (0-2 inclusive)."
                )

        return self._op_vals[operand]

    def list_operands(self):
        return [self.in_reg1, self.in_reg2, self.dst]

    def tick(self):
        self.completion_time -= 1

    @override
    def __repr__(self):
        return str(self)

    @override
    def __str__(self):
        os = ""
        if self.in_reg1 != "":
            os += f"in_reg1: {self.in_reg1} "
        if self.in_reg2 != "":
            os += f"in_reg2: {self.in_reg2} "
        if self.dst != "":
            os += f"dst: {self.dst} "
        if self.addr > -1:
            os += f"addr: {self.addr} "
        if len(os) > 0:
            os = os[:-1]
        return (
            str(self.operation)
            + " on "
            + str(os)
            + " is "
            + str(self.state)
            + " with timestamp "
            + str(self.timestamp)
            + f" ct: {self.completion_time}"
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
