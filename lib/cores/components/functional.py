from lib.errors import PimInstructionUnsupportedError
from lib.cores.instructions import Instruction, OpType
from lib.cores.components.base import BaseCore
from lib.controller.commands import Command
import numpy as np
import numpy.typing as npt
from typing import Any


def dtype_min(dtype: np.dtype) -> Any:
    if dtype.kind in "iu":
        return np.iinfo(dtype).min
    else:
        return np.finfo(dtype).min


def dtype_max(dtype: np.dtype) -> Any:
    if dtype.kind in "iu":
        return np.iinfo(dtype).max
    else:
        return np.finfo(dtype).max


def map_vec(core: BaseCore, f, ins: Instruction, dtype: npt.DTypeLike = np.int32):
    """
    A higher-order function which accepts a core, a function (f), and an instruction,
    then executes that function based on the operands passed via the instruction.
    """
    # check that both inputs are defined
    assert ins.in_reg1 != ""
    assert ins.in_reg2 != ""
    # infer that reg1 is the destination register if none is supplied
    dst = ins.dst if ins.dst != "" else ins.in_reg1
    reg0 = ins.get_state_by_operand_id(0)
    reg1 = ins.get_state_by_operand_id(1)

    for i in range(len(reg0.data)):
        reg0[i] = f(reg0[i], reg1[i])
    core.set_reg(dst, reg0)


def fold_vec(core: BaseCore, f, ins: Instruction):
    assert ins.in_reg1 != ""
    assert ins.in_reg2 != ""
    dst = ins.in_reg1 if ins.dst == "" else ins.dst
    vreg = core.get_reg(ins.in_reg2)
    acc = core.get_reg(ins.in_reg1)
    for val in np.frombuffer(vreg.data, dtype=ins.dtype):
        acc = f(acc, val)
    core.set_reg(dst, acc)


# TODO: determine whether this should implicitly assume that core.registers[0] is the return register
def red_kernel(
    core: BaseCore, cmd: Command, vector_fold_op: OpType, final_fold_op: OpType | None
):
    core.add_instruction(OpType.READ, dst=core.vec_registers[0], addr=cmd.range_1[0])
    # TODO: determine whether this should be two different functions, since the precise behavior will
    # differ based on the core implementation. This might cause performance to be qualitatively opaque.
    if len(core.vec_registers) >= 3:
        for i in range(
            int(cmd.range_1[0] / ((core.p_mem().m_gdl_width / 8)) + 1),
            int(cmd.range_1[1] / ((core.p_mem().m_gdl_width / 8))),
        ):
            target_reg = core.vec_registers[1:3][i % 2]
            # the dtype here does nothing as of the writing of this comment
            core.add_instruction(OpType.READ, addr=i, dst=target_reg, dtype=cmd.dtype)
            core.add_instruction(
                vector_fold_op,
                in_reg1=core.vec_registers[0],
                in_reg2=target_reg,
                dtype=cmd.dtype,
            )
    else:
        for i in range(
            int((cmd.range_1[0] / (core.p_mem().m_gdl_width / 8)) + 1),
            int(cmd.range_1[1] / ((core.p_mem().m_gdl_width / 8))),
        ):
            core.add_instruction(OpType.READ, addr=i)
            core.add_instruction(
                vector_fold_op, in_reg1=core.vec_registers[0], in_reg2="gdl"
            )
    if final_fold_op is not None:
        core.add_instruction(
            final_fold_op,
            in_reg1=core.registers[0],
            in_reg2=core.vec_registers[0],
            dtype=cmd.dtype,
        )
