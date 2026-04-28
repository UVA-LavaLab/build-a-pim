from lib.errors import PimInstructionUnsupportedError
from lib.cores.instructions import Instruction, OpType
from lib.cores.components.base import BaseCore
from lib.controller.commands import Command
import numpy as np
import numpy.typing as npt
import math
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

    start = ins.start_index if ins.start_index is not None else 0

    for i in range(start, len(reg0.data)):
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
    # TODO: also consider adding another algorithm which fills as many vector registers as possible then
    # load balances between adding and filling vector registers
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


def vec_vec_kernel(core: BaseCore, cmd: Command, vec_op: OpType):
    # FIXME: this does NOT account for PIM objects which wrap around the
    # address space... (low priority)
    # FIXME: there are also some explorations to be done regarding whether it
    # is faster to have a HUGE register file and load both vectors into that,
    # then accumulate between them from there safety checks
    # TODO: figure out how to programmatically relax this to allow for any
    # ratio of input to output sizes this will facilitate compression and binary
    # operations
    assert cmd.range_1[1] - cmd.range_1[0] == cmd.range_2[1] - cmd.range_2[0]
    # by the transitive property of equality, we don't need to check the last pair
    assert cmd.range_1[1] - cmd.range_1[0] == cmd.range_dst[1] - cmd.range_dst[0]

    # window size is the number of chunks we can calculate
    # without overflowing the available vector registers
    window_size_chunks = min(
        len(core.vec_registers), core.p_mem().get_config_param("n_col")
    )

    i1r = cmd.range_1
    i2r = cmd.range_2
    dstr = cmd.range_dst
    input_chunks = i1r[1] - i1r[0]

    # this function is intended to be a preemptive bounds check which masks the
    # output of out-of-bounds operations for unevenly distributed chunks (in
    # some cases, one or more banks will not have the same number of chunks as
    # the others)
    def check_bounds(addr: int):
        addr = core.p_mem().local_to_canonical_addr(core.location, addr)
        return core.p_mem().address_mapper[addr][0] != -2

    for w in range(math.ceil(input_chunks / window_size_chunks)):
        # read the batch from the first vector
        for c, b in enumerate(
            range(
                i1r[0] + w * window_size_chunks,
                min(i1r[0] + (w + 1) * window_size_chunks, i1r[1]),
            )
        ):
            if check_bounds(b):
                target_reg = core.vec_registers[c]
                core.add_instruction(
                    OpType.READ, addr=b, dst=target_reg, dtype=cmd.dtype
                )
            else:
                # read anyways, but don't modify the register values (same timing)
                core.add_instruction(OpType.READ, addr=b, dtype=cmd.dtype)

        # read the batch from the second vector and accumulate to the first register locations
        for c, b in enumerate(
            range(
                i2r[0] + w * window_size_chunks,
                min(i2r[0] + (w + 1) * window_size_chunks, i2r[1]),
            )
        ):
            if check_bounds(b):
                core.add_instruction(OpType.READ, addr=b, dtype=cmd.dtype)
                target_reg = core.vec_registers[c]
                core.add_instruction(
                    vec_op,
                    in_reg1=target_reg,
                    in_reg2="gdl",
                    dtype=cmd.dtype,
                )
            else:
                core.add_instruction(OpType.READ, addr=b, dtype=cmd.dtype)
                # append nops to match the timing of the passed instruction
                # (functionally equivalent to masking output)
                num_nops: int = core.timings[vec_op]
                for _ in range(num_nops):
                    core.add_instruction(OpType.NOP)

        # write back to core-local memory at the appropriate index
        for c, b in enumerate(
            range(
                dstr[0] + w * window_size_chunks,
                min(dstr[0] + (w + 1) * window_size_chunks, dstr[1]),
            )
        ):
            if check_bounds(b):
                core.add_instruction(
                    OpType.WRITE, in_reg1=core.vec_registers[c], addr=b, dtype=cmd.dtype
                )
            else:
                core.add_instruction(OpType.WRITE, addr=b, dtype=cmd.dtype)
