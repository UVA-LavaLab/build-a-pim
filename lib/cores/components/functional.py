from lib.errors import PimInstructionUnsupportedError
from lib.cores.instructions import Instruction, OpType as OT
from lib.cores.components.base import BaseCore
from lib.controller.commands import Command
import numpy as np
import numpy.typing as npt
import math
from typing import Any

from lib.monad import Blob


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


def conditional_jump(
    core: BaseCore, f, ins: Instruction, dtype: npt.DTypeLike = np.int32
) -> bool:
    assert ins.in_reg1 != ""
    assert ins.in_reg2 != ""
    # check that both registers are scalar registers
    assert not isinstance(ins.get_state_by_operand_id(0), Blob)
    assert not isinstance(ins.get_state_by_operand_id(1), Blob)

    dst = ins.dst if ins.dst != "" else ins.in_reg1
    reg0 = ins.get_state_by_operand_id(0)
    reg1 = ins.get_state_by_operand_id(1)
    dt = np.dtype(dtype)

    if f(dt.type(reg0), dt.type(reg1)):
        core.pc = ins.addr
        return True
    return False


def imm_operation(core: BaseCore, f, ins: Instruction):
    if ins.in_reg1 in core.registers:
        scalar_scalar(core, f, ins, dtype=ins.dtype, imm=True)
    else:
        map_scalar_vec(core, f, ins, dtype=ins.dtype, imm=True)


def scalar_scalar(
    core: BaseCore,
    f,
    ins: Instruction,
    dtype: npt.DTypeLike = np.int32,
    imm: bool = False,
):
    # check that both inputs are defined
    assert ins.in_reg1 != ""
    # check that both registers hold scalar state
    assert not isinstance(ins.get_state_by_operand_id(0), Blob)
    # in the case we're not operating on an immediate value, repeat for reg 2
    if not imm:
        assert ins.in_reg2 != ""
        assert not isinstance(ins.get_state_by_operand_id(1), Blob)

    dst = ins.dst if ins.dst != "" else ins.in_reg1
    reg0 = ins.get_state_by_operand_id(0)
    reg1 = ins.get_state_by_operand_id(1) if not imm else ins.imm
    dt = np.dtype(dtype)

    core.set_reg(dst, f(dt.type(reg0), dt.type(reg1)))


def map_scalar_vec(
    core: BaseCore,
    f,
    ins: Instruction,
    dtype: npt.DTypeLike = np.int32,
    imm: bool = False,
):
    """
    A higher-order function which accepts a core, a function (f), and an instruction,
    then executes that function based on the operands passed via the instruction.

    Note that in_reg2 of the passed instruction must be a scalar register.
    """
    # TODO: fix this for immediate values

    # check that both inputs are defined
    assert ins.in_reg1 != ""
    assert ins.in_reg2 != ""
    # check that this is a scalar register
    assert not isinstance(ins.get_state_by_operand_id(1), Blob)
    # infer that reg1 is the destination register if none is supplied
    dst = ins.dst if ins.dst != "" else ins.in_reg1
    reg0 = np.frombuffer(ins.get_state_by_operand_id(0).data, dtype=dtype)
    reg1 = ins.imm if imm else np.dtype(dtype).type(ins.get_state_by_operand_id(1))

    start = ins.start_index if ins.start_index is not None else 0

    for i in range(start, len(reg0)):
        reg0[i] = f(reg0[i], reg1)
    core.set_reg(dst, Blob(reg0))


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
    reg0 = np.frombuffer(ins.get_state_by_operand_id(0).data, dtype=dtype)
    reg1 = np.frombuffer(ins.get_state_by_operand_id(1).data, dtype=dtype)

    # early exit if both regs are length 0
    if len(reg0) == 0 and len(reg1) == 0:
        core.set_reg(dst, Blob(np.array([])))
        return
    # pass through if either length is zero
    elif len(reg0) == 0 or len(reg1) == 0:
        core.set_reg(dst, Blob(reg0 if len(reg0) > 0 else reg1))
        return

    start = ins.start_index if ins.start_index is not None else 0

    for i in range(start, min(len(reg0), len(reg1))):
        reg0[i] = f(reg0[i], reg1[i])

    core.set_reg(dst, Blob(reg0))


def fold_vec(core: BaseCore, f, ins: Instruction, dtype: npt.DTypeLike = np.int32):
    assert ins.in_reg1 != ""
    assert ins.in_reg2 != ""
    dst = ins.in_reg1 if ins.dst == "" else ins.dst
    dt = np.dtype(dtype)

    vreg = np.frombuffer(core.get_reg(ins.in_reg2).data, dtype=dtype)
    # early exit if we get an empty register
    if len(vreg) == 0:
        return
    acc = dt.type(core.get_reg(ins.in_reg1))
    for val in np.frombuffer(vreg.data, dtype=ins.dtype):
        acc = f(acc, val)
    core.set_reg(dst, acc)


# TODO: determine whether this should implicitly assume that core.registers[0] is the return register
def streamed_red_kernel(
    core: BaseCore, cmd: Command, vector_fold_op: OT, final_fold_op: OT | None
):
    prog: list[Instruction] = []
    prog.append(Instruction(OT.READ, dst=core.vec_registers[0], addr=cmd.range_1[0]))
    # TODO: determine whether this should be two different functions, since the precise behavior will
    # differ based on the core implementation. This might cause performance to be qualitatively opaque.
    # TODO: also consider adding another algorithm which fills as many vector registers as possible then
    # load balances between adding and filling vector registers
    if len(core.vec_registers) >= 3:
        for i in range(cmd.range_1[0] + 1, cmd.range_1[1]):
            target_reg = core.vec_registers[1:3][i % 2]
            # the dtype here does nothing as of the writing of this comment
            prog.append(Instruction(OT.READ, addr=i, dst=target_reg, dtype=cmd.dtype))
            prog.append(
                Instruction(
                    vector_fold_op,
                    in_reg1=core.vec_registers[0],
                    in_reg2=target_reg,
                    dtype=cmd.dtype,
                    completion_time=core.timings[vector_fold_op],
                )
            )
    else:
        for i in range(cmd.range_1[0] + 1, cmd.range_1[1]):
            prog.append(Instruction(OT.READ, addr=i))
            prog.append(
                Instruction(
                    vector_fold_op,
                    in_reg1=core.vec_registers[0],
                    in_reg2="gdl",
                    completion_time=core.timings[vector_fold_op],
                )
            )
    if final_fold_op is not None:
        prog.append(
            Instruction(
                final_fold_op,
                in_reg1=core.registers[0],
                in_reg2=core.vec_registers[0],
                dtype=cmd.dtype,
                completion_time=core.timings[final_fold_op],
            )
        )
    return prog


def streamed_vec_scalar_kernel(core: BaseCore, cmd: Command, scalar_op: OT):
    # FIXME: this does NOT account for PIM objects which wrap around the
    # address space... (low priority)
    # FIXME: there are also some explorations to be done regarding whether it
    # is faster to have a HUGE register file and load both vectors into that,
    # then accumulate between them from there safety checks
    # TODO: figure out how to programmatically relax this to allow for any
    # ratio of input to output sizes this will facilitate compression and binary
    # operations
    assert cmd.range_1[1] - cmd.range_1[0] == cmd.range_dst[1] - cmd.range_dst[0]
    # ensure that whatever scalar was passed with the command is not none
    assert cmd.scalar is not None

    # window size is the number of chunks we can calculate
    # without overflowing the available vector registers
    window_size_chunks = min(
        len(core.vec_registers), core.p_mem().get_config_param("n_col")
    )

    i1r = cmd.range_1
    dstr = cmd.range_dst
    input_chunks = i1r[1] - i1r[0]
    prog: list[Instruction] = []

    # TODO: fix this to make it an instruction which loads a scalar (MOV)
    core.set_reg(core.registers[0], cmd.scalar)

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
                prog.append(
                    Instruction(OT.READ, addr=b, dst=target_reg, dtype=cmd.dtype)
                )
            else:
                # read anyways, but don't modify the register values (same timing)
                prog.append(Instruction(OT.READ, addr=b, dtype=cmd.dtype))

        # perform the scalar operation and write back to the passed register
        for c, b in enumerate(
            range(
                i1r[0] + w * window_size_chunks,
                min(i1r[0] + (w + 1) * window_size_chunks, i1r[1]),
            )
        ):
            if check_bounds(b):
                target_reg = core.vec_registers[c]
                prog.append(
                    Instruction(
                        scalar_op,
                        in_reg1=target_reg,
                        in_reg2=core.registers[0],
                        dtype=cmd.dtype,
                    )
                )
            else:
                # append nops to match the timing of the passed instruction
                # (functionally equivalent to masking output)
                num_nops: int = core.timings[scalar_op]
                for _ in range(num_nops):
                    prog.append(Instruction(OT.NOP))

        # write back to core-local memory at the appropriate index
        for c, b in enumerate(
            range(
                dstr[0] + w * window_size_chunks,
                min(dstr[0] + (w + 1) * window_size_chunks, dstr[1]),
            )
        ):
            if check_bounds(b):
                prog.append(
                    Instruction(
                        OT.WRITE,
                        in_reg1=core.vec_registers[c],
                        addr=b,
                        dtype=cmd.dtype,
                    )
                )
            else:
                # write nothing if out of bounds (for safety, but also keeps
                # core synchronization)
                prog.append(Instruction(OT.WRITE, addr=b, dtype=cmd.dtype))

    return prog


def streamed_vec_vec_kernel(
    core: BaseCore, cmd: Command, vec_op: OT
) -> list[Instruction]:
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

    prog: list[Instruction] = []

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
                prog.append(
                    Instruction(OT.READ, addr=b, dst=target_reg, dtype=cmd.dtype)
                )
            else:
                # read anyways, but don't modify the register values (same timing)
                prog.append(Instruction(OT.READ, addr=b, dtype=cmd.dtype))

        # read the batch from the second vector and accumulate to the first register locations
        for c, b in enumerate(
            range(
                i2r[0] + w * window_size_chunks,
                min(i2r[0] + (w + 1) * window_size_chunks, i2r[1]),
            )
        ):
            if check_bounds(b):
                prog.append(Instruction(OT.READ, addr=b, dtype=cmd.dtype))
                target_reg = core.vec_registers[c]
                prog.append(
                    Instruction(
                        vec_op,
                        in_reg1=target_reg,
                        in_reg2="gdl",
                        dtype=cmd.dtype,
                    )
                )
            else:
                prog.append(Instruction(OT.READ, addr=b, dtype=cmd.dtype))
                # append nops to match the timing of the passed instruction
                # (functionally equivalent to masking output)
                num_nops: int = core.timings[vec_op]
                for _ in range(num_nops):
                    prog.append(Instruction(OT.NOP))

        # write back to core-local memory at the appropriate index
        for c, b in enumerate(
            range(
                dstr[0] + w * window_size_chunks,
                min(dstr[0] + (w + 1) * window_size_chunks, dstr[1]),
            )
        ):
            if check_bounds(b):
                prog.append(
                    Instruction(
                        OT.WRITE,
                        in_reg1=core.vec_registers[c],
                        addr=b,
                        dtype=cmd.dtype,
                    )
                )
            else:
                prog.append(Instruction(OT.WRITE, addr=b, dtype=cmd.dtype))

    return prog


def red_kernel(
    core: BaseCore, cmd: Command, vector_fold_op: OT, final_fold_op: OT | None
):
    prog: list[Instruction] = []
    # some assumptions are made for this implementation, can generalize later
    # this assumption facilitates a simple unrolling, which allows us to avoid
    # some overhead from data dependencies
    assert len(core.registers) >= 3
    assert len(core.vec_registers) >= 3
    prog.append(Instruction(OT.READ, dst=core.vec_registers[0], addr=cmd.range_1[0]))
    # the counter value, which should start at 1 since we just read the first
    # value
    prog.append(
        Instruction(
            OT.MOV,
            dst=core.registers[1],
            imm=np.int32(1),
            completion_time=core.timings[OT.MOV],
        )
    )
    # the length value
    prog.append(
        Instruction(
            OT.MOV,
            dst=core.registers[2],
            imm=np.int32(cmd.range_1[1] - cmd.range_1[0]),
            completion_time=core.timings[OT.MOV],
        )
    )

    # for now, we have no labels, so the location has to be manually defined
    loop_start: int = len(prog)
    prog.append(
        Instruction(
            OT.READ,
            addr=cmd.range_1[0],
            in_reg1=core.registers[1],
            dst=core.vec_registers[1],
            dtype=cmd.dtype,
        )
    )
    prog.append(
        Instruction(
            vector_fold_op,
            in_reg1=core.vec_registers[0],
            in_reg2=core.vec_registers[1],
            dst=core.vec_registers[0],
            dtype=cmd.dtype,
            completion_time=core.timings[vector_fold_op],
        )
    )
    prog.append(
        Instruction(
            OT.IMM_ADD,
            in_reg1=core.registers[1],
            dst=core.registers[1],
            dtype=cmd.dtype,
            imm=np.int32(1),
            completion_time=core.timings[OT.IMM_ADD],
        )
    )
    # bounds check early, jump to end of program if we're past the bounds
    prog.append(
        Instruction(
            OT.JGE,
            in_reg1=core.registers[1],
            in_reg2=core.registers[2],
            addr=len(prog) + 5,
            completion_time=core.timings[OT.JGE],
        )
    )
    prog.append(
        Instruction(
            OT.READ,
            addr=cmd.range_1[0],
            in_reg1=core.registers[1],
            dst=core.vec_registers[2],
            dtype=cmd.dtype,
        )
    )
    prog.append(
        Instruction(
            vector_fold_op,
            in_reg1=core.vec_registers[0],
            in_reg2=core.vec_registers[2],
            dst=core.vec_registers[0],
            dtype=cmd.dtype,
            completion_time=core.timings[vector_fold_op],
        )
    )
    prog.append(
        Instruction(
            OT.IMM_ADD,
            in_reg1=core.registers[1],
            dst=core.registers[1],
            imm=np.int32(1),
            completion_time=core.timings[OT.IMM_ADD],
        )
    )
    # if the counter is less than the length, jump to the beginning of the loop
    prog.append(
        Instruction(
            OT.JL,
            in_reg1=core.registers[1],
            in_reg2=core.registers[2],
            addr=loop_start,
            completion_time=core.timings[OT.JL],
        )
    )
    if final_fold_op is not None:
        prog.append(
            Instruction(
                final_fold_op,
                in_reg1=core.registers[0],
                in_reg2=core.vec_registers[0],
                dst=core.registers[0],
                completion_time=core.timings[final_fold_op],
            )
        )

    return prog


def vec_vec_kernel(core: BaseCore, cmd: Command, vec_op: OT) -> list[Instruction]:
    assert cmd.range_1[1] - cmd.range_1[0] == cmd.range_2[1] - cmd.range_2[0]
    # by the transitive property of equality, we don't need to check the last pair
    assert cmd.range_1[1] - cmd.range_1[0] == cmd.range_dst[1] - cmd.range_dst[0]

    def sel_window(n_vreg: int, allocation_len: int, n_cols: int):
        # TODO: is there a more efficient way to compile this?
        max_regs: int = min(n_vreg, allocation_len, n_cols)
        # linear scan
        for w in range(max_regs, 0, -1):
            if allocation_len % w == 0:
                return w
        return 1

    n_cols: int = core.p_mem().get_config_param("n_col")
    n_vreg: int = len(core.vec_registers)
    allocation_len: int = cmd.range_1[1] - cmd.range_1[0]

    # window size is the number of chunks we can calculate
    # without overflowing the available vector registers

    # we also constrain the window size to be an integer divisor of the column
    # size, since non-integer divisors will perform out of bounds accesses
    # without additional costly runtime checks (a JL/JG after every access)
    window_size_chunks = sel_window(n_vreg, allocation_len, n_cols)

    i1r = cmd.range_1
    i2r = cmd.range_2
    dstr = cmd.range_dst
    input_chunks = i1r[1] - i1r[0]

    prog: list[Instruction] = []

    prog.append(
        Instruction(
            OT.MOV,
            dst=core.registers[0],
            imm=np.int32(0),
            completion_time=core.timings[OT.MOV],
        )
    )

    prog.append(
        Instruction(
            OT.MOV,
            dst=core.registers[1],
            imm=np.int32(input_chunks),
            completion_time=core.timings[OT.MOV],
        )
    )

    loop = len(prog)

    # we want to unroll over the set of used registers
    # read the batch from the first vector, we will add the window size as an immediate at the end of each loop
    for c, b in enumerate(
        range(
            i1r[0],
            min(i1r[0] + window_size_chunks, i1r[1]),
        )
    ):
        target_reg = core.vec_registers[c]
        prog.append(
            Instruction(
                OT.READ,
                addr=b,
                in_reg1=core.registers[0],
                dst=target_reg,
                dtype=cmd.dtype,
            )
        )

    # read the batch from the second vector and accumulate to the first register locations
    for c, b in enumerate(
        range(
            i2r[0],
            min(i2r[0] + window_size_chunks, i2r[1]),
        )
    ):
        prog.append(
            Instruction(OT.READ, in_reg1=core.registers[0], addr=b, dtype=cmd.dtype)
        )
        target_reg = core.vec_registers[c]
        prog.append(
            Instruction(
                vec_op,
                in_reg1=target_reg,
                in_reg2="gdl",
                dtype=cmd.dtype,
            )
        )

    # write back to core-local memory at the appropriate index
    for c, b in enumerate(
        range(
            dstr[0],
            min(dstr[0] + window_size_chunks, dstr[1]),
        )
    ):
        prog.append(
            Instruction(
                OT.WRITE,
                in_reg1=core.vec_registers[c],
                in_reg2=core.registers[0],
                addr=b,
                dtype=cmd.dtype,
            )
        )

    prog.append(
        Instruction(
            OT.IMM_ADD,
            in_reg1=core.registers[0],
            dst=core.registers[0],
            imm=np.int32(window_size_chunks),
            dtype=cmd.dtype,
            completion_time=core.timings[OT.IMM_ADD],
        )
    )

    prog.append(
        Instruction(
            OT.JL,
            in_reg1=core.registers[0],
            in_reg2=core.registers[1],
            addr=loop,
            dtype=cmd.dtype,
            completion_time=core.timings[OT.JL],
        )
    )

    return prog


def vec_scalar_kernel(core: BaseCore, cmd: Command, scalar_op: OT):
    assert cmd.scalar is not None

    def sel_window(n_vreg: int, allocation_len: int, n_cols: int):
        # TODO: is there a more efficient way to compile this?
        max_regs: int = min(n_vreg, allocation_len, n_cols)
        # linear scan
        for w in range(max_regs, 0, -1):
            if allocation_len % w == 0:
                return w
        return 1

    n_cols: int = core.p_mem().get_config_param("n_col")
    n_vreg: int = len(core.vec_registers)
    allocation_len: int = cmd.range_1[1] - cmd.range_1[0]

    # window size is the number of chunks we can calculate
    # without overflowing the available vector registers

    # we also constrain the window size to be an integer divisor of the column
    # size, since non-integer divisors will perform out of bounds accesses
    # without additional costly runtime checks (a JL/JG after every access)
    window_size_chunks = sel_window(n_vreg, allocation_len, n_cols)

    i1r = cmd.range_1
    i2r = cmd.range_2
    dstr = cmd.range_dst
    input_chunks = i1r[1] - i1r[0]

    prog: list[Instruction] = []

    prog.append(
        Instruction(
            OT.MOV,
            dst=core.registers[2],
            imm=cmd.scalar,
            completion_time=core.timings[OT.MOV],
        )
    )

    prog.append(
        Instruction(
            OT.MOV,
            dst=core.registers[0],
            imm=np.int32(0),
            completion_time=core.timings[OT.MOV],
        )
    )

    prog.append(
        Instruction(
            OT.MOV,
            dst=core.registers[1],
            imm=np.int32(input_chunks),
            completion_time=core.timings[OT.MOV],
        )
    )

    loop = len(prog)

    # we want to unroll over the set of used registers
    # read the batch from the first vector, we will add the window size as an immediate at the end of each loop
    for c, b in enumerate(
        range(
            i1r[0],
            min(i1r[0] + window_size_chunks, i1r[1]),
        )
    ):
        target_reg = core.vec_registers[c]
        prog.append(
            Instruction(
                OT.READ,
                addr=b,
                in_reg1=core.registers[0],
                dst=target_reg,
                dtype=cmd.dtype,
            )
        )

    # read the batch from the second vector and accumulate to the first register locations
    for c, b in enumerate(
        range(
            i1r[0],
            min(i1r[0] + window_size_chunks, i1r[1]),
        )
    ):
        target_reg = core.vec_registers[c]
        prog.append(
            Instruction(
                scalar_op,
                in_reg1=target_reg,
                in_reg2=core.registers[2],
                dtype=cmd.dtype,
                completion_time=core.timings[scalar_op],
            )
        )

    # write back to core-local memory at the appropriate index
    for c, b in enumerate(
        range(
            dstr[0],
            min(dstr[0] + window_size_chunks, dstr[1]),
        )
    ):
        prog.append(
            Instruction(
                OT.WRITE,
                in_reg1=core.vec_registers[c],
                in_reg2=core.registers[0],
                addr=b,
                dtype=cmd.dtype,
            )
        )

    prog.append(
        Instruction(
            OT.IMM_ADD,
            in_reg1=core.registers[0],
            dst=core.registers[0],
            imm=np.int32(window_size_chunks),
            dtype=cmd.dtype,
            completion_time=core.timings[OT.IMM_ADD],
        )
    )

    prog.append(
        Instruction(
            OT.JL,
            in_reg1=core.registers[0],
            in_reg2=core.registers[1],
            addr=loop,
            dtype=cmd.dtype,
            completion_time=core.timings[OT.JL],
        )
    )

    return prog
