# These functions have to be duplicated for both transparent testing and to
# give proper type hints. Half of these functions don't type check anyways, but
# we want to allow as many of them as physically possible to type check.

from lib.memsys import MemSystem
from lib.containers import Ptr
from lib.cores.ins_stream_bank_simd import Core as StreamingCore
from lib.cores.bank_simd_scratch import Core
from lib.controller.commands import Command, CommandType
from lib.address.allocation import pim_device_place_data
from typing import Any
import numpy as np
import numpy.typing as npt
import random


def setup_device(
    coretype: type = StreamingCore,
) -> tuple[MemSystem, list[StreamingCore] | list[Core]]:
    mem = MemSystem("./dramsim3/configs/HBM2_8Gb_x128.ini", ".", nd_log=True)
    cores = [
        coretype((c, r, bg, b), Ptr(mem))
        for c in range(mem.num_channels)
        for r in range(mem.num_ranks)
        for bg in range(mem.num_bankgroups_per_rank)
        for b in range(mem.num_banks_per_group)
    ]

    return mem, cores


def gen_data(
    vec_len: int = 16384, dtype: npt.DTypeLike = np.int32
) -> tuple[npt.NDArray[np.generic], npt.NDArray[np.generic], npt.NDArray[np.generic]]:
    l1: npt.NDArray[np.generic] = np.array(
        [random.randint(1, 5) for _ in range(vec_len)], dtype=dtype
    )
    l2: npt.NDArray[np.generic] = np.array(
        [random.randint(1, 5) for _ in range(vec_len)], dtype=dtype
    )
    dst: npt.NDArray[np.generic] = np.zeros(vec_len, dtype=dtype)
    return l1, l2, dst


# def place_data(mem: MemSystem, cores: list[StreamingCore], data: npt.NDArray[np.generic], addr: int = 0):
def start_command(
    cores: list[StreamingCore] | list[Core],
    cmdtype: CommandType,
    range1: tuple[int, int] | None,
    range2: tuple[int, int] | None,
    dst: tuple[int, int] | None,
    scalar: Any | None = None,
    dtype: npt.DTypeLike = np.int32,
):
    for core in cores:
        core.tick(
            cmd=Command(
                cmdtype,
                range1[0] if range1 is not None else 0,
                range1[1] if range1 is not None else 0,
                range2[0] if range2 is not None else 0,
                range2[1] if range2 is not None else 0,
                dst[0] if dst is not None else 0,
                dst[1] if dst is not None else 0,
                scalar=scalar if scalar is not None else None,
                dtype=dtype,
            )
        )


def test_vadd_streaming_float64():
    vec_len: int = 16384

    mem, cores = setup_device()
    l1, l2, dst = gen_data(vec_len, dtype=np.float64)

    id_l1, r_op1 = pim_device_place_data(mem, cores, l1, 0)
    id_l2, r_op2 = pim_device_place_data(mem, cores, l2, r_op1[1])
    id_dst, r_dst = pim_device_place_data(mem, cores, dst, r_op2[1])

    start_command(cores, CommandType.PIM_ADD, r_op1, r_op2, r_dst, dtype=np.float64)

    i = 0
    while True:
        all_done = True
        for j, core in enumerate(cores):
            core.tick()
            all_done = not (
                len(core.instruction_queue) > 0
                or not core.pipeline.is_empty()
                or i < 100
            )
        mem.tick()
        i += 1
        if all_done:
            break

    # use r_dst to find outputs
    outputs_arr: npt.NDArray[np.float64] = np.frombuffer(
        mem.stored_data_structures[id_dst].data_structure, dtype=np.float64
    )
    all_match = np.all(
        outputs_arr
        == (np.frombuffer(l1, dtype=np.float64) + np.frombuffer(l2, dtype=np.float64))
    )
    if not all_match:
        print("output", outputs_arr)
        print("expected", l1 + l2)
        print(f"input1, {l1}")
        print(f"input2, {l2}")
    assert all_match


def test_vadd_riscv_float64():
    vec_len: int = 16384

    mem, cores = setup_device(coretype=Core)
    l1, l2, dst = gen_data(vec_len, dtype=np.float64)

    id_l1, r_op1 = pim_device_place_data(mem, cores, l1, 0)
    id_l2, r_op2 = pim_device_place_data(mem, cores, l2, r_op1[1])
    id_dst, r_dst = pim_device_place_data(mem, cores, dst, r_op2[1])

    start_command(cores, CommandType.PIM_ADD, r_op1, r_op2, r_dst, dtype=np.float64)

    i = 0
    while True:
        all_done = True
        for j, core in enumerate(cores):
            core.tick()
            all_done = not (
                len(core.instruction_queue) > 0
                or not core.pipeline.is_empty()
                or i < 100
            )
        mem.tick()
        i += 1
        if all_done:
            break

    # use r_dst to find outputs
    outputs_arr: npt.NDArray[np.float64] = np.frombuffer(
        mem.stored_data_structures[id_dst].data_structure, dtype=np.float64
    )
    all_match = np.all(
        outputs_arr
        == (np.frombuffer(l1, dtype=np.float64) + np.frombuffer(l2, dtype=np.float64))
    )
    if not all_match:
        print("output", outputs_arr)
        print("expected", l1 + l2)
        print(f"input1, {l1}")
        print(f"input2, {l2}")
    assert all_match


def test_vadd_streaming_float16():
    vec_len: int = 16384

    mem, cores = setup_device()
    l1, l2, dst = gen_data(vec_len, dtype=np.float16)

    id_l1, r_op1 = pim_device_place_data(mem, cores, l1, 0)
    id_l2, r_op2 = pim_device_place_data(mem, cores, l2, r_op1[1])
    id_dst, r_dst = pim_device_place_data(mem, cores, dst, r_op2[1])

    start_command(cores, CommandType.PIM_ADD, r_op1, r_op2, r_dst, dtype=np.float16)

    i = 0
    while True:
        all_done = True
        for j, core in enumerate(cores):
            core.tick()
            all_done = not (
                len(core.instruction_queue) > 0
                or not core.pipeline.is_empty()
                or i < 100
            )
        mem.tick()
        i += 1
        if all_done:
            break

    # use r_dst to find outputs
    outputs_arr: npt.NDArray[np.float16] = np.frombuffer(
        mem.stored_data_structures[id_dst].data_structure, dtype=np.float16
    )
    all_match = np.all(
        outputs_arr
        == (np.frombuffer(l1, dtype=np.float16) + np.frombuffer(l2, dtype=np.float16))
    )
    if not all_match:
        print("output", outputs_arr)
        print("expected", l1 + l2)
        print(f"input1, {l1}")
        print(f"input2, {l2}")
    assert all_match


def test_vadd_riscv_float16():
    vec_len: int = 16384

    mem, cores = setup_device(coretype=Core)
    l1, l2, dst = gen_data(vec_len, dtype=np.float16)

    id_l1, r_op1 = pim_device_place_data(mem, cores, l1, 0)
    id_l2, r_op2 = pim_device_place_data(mem, cores, l2, r_op1[1])
    id_dst, r_dst = pim_device_place_data(mem, cores, dst, r_op2[1])

    start_command(cores, CommandType.PIM_ADD, r_op1, r_op2, r_dst, dtype=np.float16)

    i = 0
    while True:
        all_done = True
        for j, core in enumerate(cores):
            core.tick()
            all_done = not (
                len(core.instruction_queue) > 0
                or not core.pipeline.is_empty()
                or i < 100
            )
        mem.tick()
        i += 1
        if all_done:
            break

    # use r_dst to find outputs
    outputs_arr: npt.NDArray[np.float16] = np.frombuffer(
        mem.stored_data_structures[id_dst].data_structure, dtype=np.float16
    )
    all_match = np.all(
        outputs_arr
        == (np.frombuffer(l1, dtype=np.float16) + np.frombuffer(l2, dtype=np.float16))
    )
    if not all_match:
        print("output", outputs_arr)
        print("expected", l1 + l2)
        print(f"input1, {l1}")
        print(f"input2, {l2}")
    assert all_match


def test_vadd_streaming_float128():
    vec_len: int = 16384

    mem, cores = setup_device()
    l1, l2, dst = gen_data(vec_len, dtype=np.float128)

    id_l1, r_op1 = pim_device_place_data(mem, cores, l1, 0)
    id_l2, r_op2 = pim_device_place_data(mem, cores, l2, r_op1[1])
    id_dst, r_dst = pim_device_place_data(mem, cores, dst, r_op2[1])

    start_command(cores, CommandType.PIM_ADD, r_op1, r_op2, r_dst, dtype=np.float128)

    i = 0
    while True:
        all_done = True
        for j, core in enumerate(cores):
            core.tick()
            all_done = not (
                len(core.instruction_queue) > 0
                or not core.pipeline.is_empty()
                or i < 100
            )
        mem.tick()
        i += 1
        if all_done:
            break

    # use r_dst to find outputs
    outputs_arr: npt.NDArray[np.float128] = np.frombuffer(
        mem.stored_data_structures[id_dst].data_structure, dtype=np.float128
    )
    all_match = np.all(
        outputs_arr
        == (np.frombuffer(l1, dtype=np.float128) + np.frombuffer(l2, dtype=np.float128))
    )
    if not all_match:
        print("output", outputs_arr)
        print("expected", l1 + l2)
        print(f"input1, {l1}")
        print(f"input2, {l2}")
    assert all_match


def test_vadd_riscv_float128():
    vec_len: int = 16384

    mem, cores = setup_device(coretype=Core)
    l1, l2, dst = gen_data(vec_len, dtype=np.float128)

    id_l1, r_op1 = pim_device_place_data(mem, cores, l1, 0)
    id_l2, r_op2 = pim_device_place_data(mem, cores, l2, r_op1[1])
    id_dst, r_dst = pim_device_place_data(mem, cores, dst, r_op2[1])

    start_command(cores, CommandType.PIM_ADD, r_op1, r_op2, r_dst, dtype=np.float128)

    i = 0
    while True:
        all_done = True
        for j, core in enumerate(cores):
            core.tick()
            all_done = not (
                len(core.instruction_queue) > 0
                or not core.pipeline.is_empty()
                or i < 100
            )
        mem.tick()
        i += 1
        if all_done:
            break

    # use r_dst to find outputs
    outputs_arr: npt.NDArray[np.float128] = np.frombuffer(
        mem.stored_data_structures[id_dst].data_structure, dtype=np.float128
    )
    all_match = np.all(
        outputs_arr
        == (np.frombuffer(l1, dtype=np.float128) + np.frombuffer(l2, dtype=np.float128))
    )
    if not all_match:
        print("output", outputs_arr)
        print("expected", l1 + l2)
        print(f"input1, {l1}")
        print(f"input2, {l2}")
    assert all_match


def test_vadd_streaming_int64():
    vec_len: int = 16384

    mem, cores = setup_device()
    l1, l2, dst = gen_data(vec_len, dtype=np.int64)

    id_l1, r_op1 = pim_device_place_data(mem, cores, l1, 0)
    id_l2, r_op2 = pim_device_place_data(mem, cores, l2, r_op1[1])
    id_dst, r_dst = pim_device_place_data(mem, cores, dst, r_op2[1])

    start_command(cores, CommandType.PIM_ADD, r_op1, r_op2, r_dst, dtype=np.int64)

    i = 0
    while True:
        all_done = True
        for j, core in enumerate(cores):
            core.tick()
            all_done = not (
                len(core.instruction_queue) > 0
                or not core.pipeline.is_empty()
                or i < 100
            )
        mem.tick()
        i += 1
        if all_done:
            break

    # use r_dst to find outputs
    outputs_arr: npt.NDArray[np.int64] = np.frombuffer(
        mem.stored_data_structures[id_dst].data_structure, dtype=np.int64
    )
    all_match = np.all(
        outputs_arr
        == (np.frombuffer(l1, dtype=np.int64) + np.frombuffer(l2, dtype=np.int64))
    )
    if not all_match:
        print("output", outputs_arr)
        print("expected", l1 + l2)
        print(f"input1, {l1}")
        print(f"input2, {l2}")
    assert all_match


def test_vadd_riscv_int64():
    vec_len: int = 16384

    mem, cores = setup_device(coretype=Core)
    l1, l2, dst = gen_data(vec_len, dtype=np.int64)

    id_l1, r_op1 = pim_device_place_data(mem, cores, l1, 0)
    id_l2, r_op2 = pim_device_place_data(mem, cores, l2, r_op1[1])
    id_dst, r_dst = pim_device_place_data(mem, cores, dst, r_op2[1])

    start_command(cores, CommandType.PIM_ADD, r_op1, r_op2, r_dst, dtype=np.int64)

    i = 0
    while True:
        all_done = True
        for j, core in enumerate(cores):
            core.tick()
            all_done = not (
                len(core.instruction_queue) > 0
                or not core.pipeline.is_empty()
                or i < 100
            )
        mem.tick()
        i += 1
        if all_done:
            break

    # use r_dst to find outputs
    outputs_arr: npt.NDArray[np.int64] = np.frombuffer(
        mem.stored_data_structures[id_dst].data_structure, dtype=np.int64
    )
    all_match = np.all(
        outputs_arr
        == (np.frombuffer(l1, dtype=np.int64) + np.frombuffer(l2, dtype=np.int64))
    )
    if not all_match:
        print("output", outputs_arr)
        print("expected", l1 + l2)
        print(f"input1, {l1}")
        print(f"input2, {l2}")
    assert all_match

def test_vadd_streaming_uint32():
    vec_len: int = 16384

    mem, cores = setup_device()
    l1, l2, dst = gen_data(vec_len, dtype=np.uint32)

    id_l1, r_op1 = pim_device_place_data(mem, cores, l1, 0)
    id_l2, r_op2 = pim_device_place_data(mem, cores, l2, r_op1[1])
    id_dst, r_dst = pim_device_place_data(mem, cores, dst, r_op2[1])

    start_command(cores, CommandType.PIM_ADD, r_op1, r_op2, r_dst, dtype=np.uint32)

    i = 0
    while True:
        all_done = True
        for j, core in enumerate(cores):
            core.tick()
            all_done = not (
                len(core.instruction_queue) > 0
                or not core.pipeline.is_empty()
                or i < 100
            )
        mem.tick()
        i += 1
        if all_done:
            break

    # use r_dst to find outputs
    outputs_arr: npt.NDArray[np.uint32] = np.frombuffer(
        mem.stored_data_structures[id_dst].data_structure, dtype=np.uint32
    )
    all_match = np.all(
        outputs_arr
        == (np.frombuffer(l1, dtype=np.uint32) + np.frombuffer(l2, dtype=np.uint32))
    )
    if not all_match:
        print("output", outputs_arr)
        print("expected", l1 + l2)
        print(f"input1, {l1}")
        print(f"input2, {l2}")
    assert all_match


def test_vadd_riscv_uint32():
    vec_len: int = 16384

    mem, cores = setup_device(coretype=Core)
    l1, l2, dst = gen_data(vec_len, dtype=np.uint32)

    id_l1, r_op1 = pim_device_place_data(mem, cores, l1, 0)
    id_l2, r_op2 = pim_device_place_data(mem, cores, l2, r_op1[1])
    id_dst, r_dst = pim_device_place_data(mem, cores, dst, r_op2[1])

    start_command(cores, CommandType.PIM_ADD, r_op1, r_op2, r_dst, dtype=np.uint32)

    i = 0
    while True:
        all_done = True
        for j, core in enumerate(cores):
            core.tick()
            all_done = not (
                len(core.instruction_queue) > 0
                or not core.pipeline.is_empty()
                or i < 100
            )
        mem.tick()
        i += 1
        if all_done:
            break

    # use r_dst to find outputs
    outputs_arr: npt.NDArray[np.uint32] = np.frombuffer(
        mem.stored_data_structures[id_dst].data_structure, dtype=np.uint32
    )
    all_match = np.all(
        outputs_arr
        == (np.frombuffer(l1, dtype=np.uint32) + np.frombuffer(l2, dtype=np.uint32))
    )
    if not all_match:
        print("output", outputs_arr)
        print("expected", l1 + l2)
        print(f"input1, {l1}")
        print(f"input2, {l2}")
    assert all_match

def test_vadd_streaming_uint16():
    vec_len: int = 16384

    mem, cores = setup_device()
    l1, l2, dst = gen_data(vec_len, dtype=np.uint16)

    id_l1, r_op1 = pim_device_place_data(mem, cores, l1, 0)
    id_l2, r_op2 = pim_device_place_data(mem, cores, l2, r_op1[1])
    id_dst, r_dst = pim_device_place_data(mem, cores, dst, r_op2[1])

    start_command(cores, CommandType.PIM_ADD, r_op1, r_op2, r_dst, dtype=np.uint16)

    i = 0
    while True:
        all_done = True
        for j, core in enumerate(cores):
            core.tick()
            all_done = not (
                len(core.instruction_queue) > 0
                or not core.pipeline.is_empty()
                or i < 100
            )
        mem.tick()
        i += 1
        if all_done:
            break

    # use r_dst to find outputs
    outputs_arr: npt.NDArray[np.uint16] = np.frombuffer(
        mem.stored_data_structures[id_dst].data_structure, dtype=np.uint16
    )
    all_match = np.all(
        outputs_arr
        == (np.frombuffer(l1, dtype=np.uint16) + np.frombuffer(l2, dtype=np.uint16))
    )
    if not all_match:
        print("output", outputs_arr)
        print("expected", l1 + l2)
        print(f"input1, {l1}")
        print(f"input2, {l2}")
    assert all_match


def test_vadd_riscv_uint16():
    vec_len: int = 16384

    mem, cores = setup_device(coretype=Core)
    l1, l2, dst = gen_data(vec_len, dtype=np.uint16)

    id_l1, r_op1 = pim_device_place_data(mem, cores, l1, 0)
    id_l2, r_op2 = pim_device_place_data(mem, cores, l2, r_op1[1])
    id_dst, r_dst = pim_device_place_data(mem, cores, dst, r_op2[1])

    start_command(cores, CommandType.PIM_ADD, r_op1, r_op2, r_dst, dtype=np.uint16)

    i = 0
    while True:
        all_done = True
        for j, core in enumerate(cores):
            core.tick()
            all_done = not (
                len(core.instruction_queue) > 0
                or not core.pipeline.is_empty()
                or i < 100
            )
        mem.tick()
        i += 1
        if all_done:
            break

    # use r_dst to find outputs
    outputs_arr: npt.NDArray[np.uint16] = np.frombuffer(
        mem.stored_data_structures[id_dst].data_structure, dtype=np.uint16
    )
    all_match = np.all(
        outputs_arr
        == (np.frombuffer(l1, dtype=np.uint16) + np.frombuffer(l2, dtype=np.uint16))
    )
    if not all_match:
        print("output", outputs_arr)
        print("expected", l1 + l2)
        print(f"input1, {l1}")
        print(f"input2, {l2}")
    assert all_match

def test_vadd_streaming_int16():
    vec_len: int = 16384

    mem, cores = setup_device()
    l1, l2, dst = gen_data(vec_len, dtype=np.int16)

    id_l1, r_op1 = pim_device_place_data(mem, cores, l1, 0)
    id_l2, r_op2 = pim_device_place_data(mem, cores, l2, r_op1[1])
    id_dst, r_dst = pim_device_place_data(mem, cores, dst, r_op2[1])

    start_command(cores, CommandType.PIM_ADD, r_op1, r_op2, r_dst, dtype=np.int16)

    i = 0
    while True:
        all_done = True
        for j, core in enumerate(cores):
            core.tick()
            all_done = not (
                len(core.instruction_queue) > 0
                or not core.pipeline.is_empty()
                or i < 100
            )
        mem.tick()
        i += 1
        if all_done:
            break

    # use r_dst to find outputs
    outputs_arr: npt.NDArray[np.int16] = np.frombuffer(
        mem.stored_data_structures[id_dst].data_structure, dtype=np.int16
    )
    all_match = np.all(
        outputs_arr
        == (np.frombuffer(l1, dtype=np.int16) + np.frombuffer(l2, dtype=np.int16))
    )
    if not all_match:
        print("output", outputs_arr)
        print("expected", l1 + l2)
        print(f"input1, {l1}")
        print(f"input2, {l2}")
    assert all_match


def test_vadd_riscv_int16():
    vec_len: int = 16384

    mem, cores = setup_device(coretype=Core)
    l1, l2, dst = gen_data(vec_len, dtype=np.int16)

    id_l1, r_op1 = pim_device_place_data(mem, cores, l1, 0)
    id_l2, r_op2 = pim_device_place_data(mem, cores, l2, r_op1[1])
    id_dst, r_dst = pim_device_place_data(mem, cores, dst, r_op2[1])

    start_command(cores, CommandType.PIM_ADD, r_op1, r_op2, r_dst, dtype=np.int16)

    i = 0
    while True:
        all_done = True
        for j, core in enumerate(cores):
            core.tick()
            all_done = not (
                len(core.instruction_queue) > 0
                or not core.pipeline.is_empty()
                or i < 100
            )
        mem.tick()
        i += 1
        if all_done:
            break

    # use r_dst to find outputs
    outputs_arr: npt.NDArray[np.int16] = np.frombuffer(
        mem.stored_data_structures[id_dst].data_structure, dtype=np.int16
    )
    all_match = np.all(
        outputs_arr
        == (np.frombuffer(l1, dtype=np.int16) + np.frombuffer(l2, dtype=np.int16))
    )
    if not all_match:
        print("output", outputs_arr)
        print("expected", l1 + l2)
        print(f"input1, {l1}")
        print(f"input2, {l2}")
    assert all_match


def test_vadd_riscv_runtime_differs_by_type_size():
    vec_len: int = 16384

    mem, cores = setup_device(coretype=Core)
    l1, l2, dst = gen_data(vec_len, dtype=np.int16)

    id_l1, r_op1 = pim_device_place_data(mem, cores, l1, 0)
    id_l2, r_op2 = pim_device_place_data(mem, cores, l2, r_op1[1])
    id_dst, r_dst = pim_device_place_data(mem, cores, dst, r_op2[1])

    start_command(cores, CommandType.PIM_ADD, r_op1, r_op2, r_dst, dtype=np.int16)

    i = 0
    while True:
        all_done = True
        for j, core in enumerate(cores):
            core.tick()
            all_done = not (
                len(core.instruction_queue) > 0
                or not core.pipeline.is_empty()
                or i < 100
            )
        mem.tick()
        i += 1
        if all_done:
            break
    
    runtime_int16 = mem.cycle

    vec_len: int = 16384

    mem, cores = setup_device(coretype=Core)
    l1, l2, dst = gen_data(vec_len, dtype=np.int32)

    id_l1, r_op1 = pim_device_place_data(mem, cores, l1, 0)
    id_l2, r_op2 = pim_device_place_data(mem, cores, l2, r_op1[1])
    id_dst, r_dst = pim_device_place_data(mem, cores, dst, r_op2[1])

    start_command(cores, CommandType.PIM_ADD, r_op1, r_op2, r_dst, dtype=np.int32)

    i = 0
    while True:
        all_done = True
        for j, core in enumerate(cores):
            core.tick()
            all_done = not (
                len(core.instruction_queue) > 0
                or not core.pipeline.is_empty()
                or i < 100
            )
        mem.tick()
        i += 1
        if all_done:
            break

    runtime_int32 = mem.cycle

    assert runtime_int16 < runtime_int32

def test_vadd_streaming_runtime_differs_by_type_size():
    vec_len: int = 16384

    mem, cores = setup_device(coretype=StreamingCore)
    l1, l2, dst = gen_data(vec_len, dtype=np.int16)

    id_l1, r_op1 = pim_device_place_data(mem, cores, l1, 0)
    id_l2, r_op2 = pim_device_place_data(mem, cores, l2, r_op1[1])
    id_dst, r_dst = pim_device_place_data(mem, cores, dst, r_op2[1])

    start_command(cores, CommandType.PIM_ADD, r_op1, r_op2, r_dst, dtype=np.int16)

    i = 0
    while True:
        all_done = True
        for j, core in enumerate(cores):
            core.tick()
            all_done = not (
                len(core.instruction_queue) > 0
                or not core.pipeline.is_empty()
                or i < 100
            )
        mem.tick()
        i += 1
        if all_done:
            break
    
    runtime_int16 = mem.cycle

    vec_len: int = 16384

    mem, cores = setup_device(coretype=StreamingCore)
    l1, l2, dst = gen_data(vec_len, dtype=np.int32)

    id_l1, r_op1 = pim_device_place_data(mem, cores, l1, 0)
    id_l2, r_op2 = pim_device_place_data(mem, cores, l2, r_op1[1])
    id_dst, r_dst = pim_device_place_data(mem, cores, dst, r_op2[1])

    start_command(cores, CommandType.PIM_ADD, r_op1, r_op2, r_dst, dtype=np.int32)

    i = 0
    while True:
        all_done = True
        for j, core in enumerate(cores):
            core.tick()
            all_done = not (
                len(core.instruction_queue) > 0
                or not core.pipeline.is_empty()
                or i < 100
            )
        mem.tick()
        i += 1
        if all_done:
            break

    runtime_int32 = mem.cycle

    assert runtime_int16 < runtime_int32
