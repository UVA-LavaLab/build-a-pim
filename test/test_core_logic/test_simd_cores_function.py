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


def setup_device(coretype: type = StreamingCore) -> tuple[MemSystem, list[StreamingCore] | list[Core]]:
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
    vec_len: int = 16384,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    l1: npt.NDArray[np.int32] = np.array(
        [random.randint(1, 5) for _ in range(vec_len)], dtype=np.int32
    )
    l2: npt.NDArray[np.int32] = np.array(
        [random.randint(1, 5) for _ in range(vec_len)], dtype=np.int32
    )
    dst: npt.NDArray[np.int32] = np.zeros(vec_len, dtype=np.int32)
    return l1, l2, dst


# def place_data(mem: MemSystem, cores: list[StreamingCore], data: npt.NDArray[np.generic], addr: int = 0):
def start_command(
    cores: list[StreamingCore] | list[Core],
    cmdtype: CommandType,
    range1: tuple[int, int] | None,
    range2: tuple[int, int] | None,
    dst: tuple[int, int] | None,
    scalar: Any | None = None,
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
                dtype=np.int32,
            )
        )


def test_vadd_streaming():
    vec_len:int = 16384

    mem, cores = setup_device()
    l1, l2, dst = gen_data(vec_len)

    id_l1, r_op1 = pim_device_place_data(mem, cores, l1, 0)
    id_l2, r_op2 = pim_device_place_data(mem, cores, l2, r_op1[1])
    id_dst, r_dst = pim_device_place_data(mem, cores, dst, r_op2[1])

    start_command(cores, CommandType.PIM_ADD, r_op1, r_op2, r_dst)

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
    outputs_arr: npt.NDArray[np.int32] = np.frombuffer(
        mem.stored_data_structures[id_dst].data_structure, dtype=np.int32
    )
    all_match = np.all(outputs_arr == (l1 + l2))
    if not all_match:
        print("output", outputs_arr)
        print("expected", l1 + l2)
        print(f"input1, {l1}")
        print(f"input2, {l2}")
    assert all_match


def test_vadd_riscv():
    vec_len:int = 16384

    mem, cores = setup_device(coretype=Core)
    l1, l2, dst = gen_data(vec_len)

    id_l1, r_op1 = pim_device_place_data(mem, cores, l1, 0)
    id_l2, r_op2 = pim_device_place_data(mem, cores, l2, r_op1[1])
    id_dst, r_dst = pim_device_place_data(mem, cores, dst, r_op2[1])

    start_command(cores, CommandType.PIM_ADD, r_op1, r_op2, r_dst)

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
    outputs_arr: npt.NDArray[np.int32] = np.frombuffer(
        mem.stored_data_structures[id_dst].data_structure, dtype=np.int32
    )
    all_match = np.all(outputs_arr == (l1 + l2))
    if not all_match:
        print("output", outputs_arr)
        print("expected", l1 + l2)
        print(f"input1, {l1}")
        print(f"input2, {l2}")
    assert all_match


def test_scalar_vadd_streaming():
    vec_len:int = 16384

    mem, cores = setup_device()
    l1, l2, dst = gen_data(vec_len)

    id_l1, r_op1 = pim_device_place_data(mem, cores, l1, 0)
    id_l2, r_op2 = pim_device_place_data(mem, cores, l2, r_op1[1])
    id_dst, r_dst = pim_device_place_data(mem, cores, dst, r_op2[1])

    start_command(cores, CommandType.PIM_SCALAR_ADD, r_op1, None, r_dst, scalar=12)

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
    outputs_arr: npt.NDArray[np.int32] = np.frombuffer(
        mem.stored_data_structures[id_dst].data_structure, dtype=np.int32
    )
    all_match = np.all(outputs_arr == (l1 + 12))
    if not all_match:
        print("output", outputs_arr)
        print("expected", l1 + l2)
        print(f"input1, {l1}")
        print(f"input2, {l2}")
    assert all_match


def test_scalar_vadd_riscv():
    vec_len:int = 16384

    mem, cores = setup_device(Core)
    l1, l2, dst = gen_data(vec_len)

    id_l1, r_op1 = pim_device_place_data(mem, cores, l1, 0)
    id_l2, r_op2 = pim_device_place_data(mem, cores, l2, r_op1[1])
    id_dst, r_dst = pim_device_place_data(mem, cores, dst, r_op2[1])

    start_command(cores, CommandType.PIM_SCALAR_ADD, r_op1, None, r_dst, scalar=12)

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
    outputs_arr: npt.NDArray[np.int32] = np.frombuffer(
        mem.stored_data_structures[id_dst].data_structure, dtype=np.int32
    )
    all_match = np.all(outputs_arr == (l1 + 12))
    if not all_match:
        print("output", outputs_arr)
        print("expected", l1 + l2)
        print(f"input1, {l1}")
        print(f"input2, {l2}")
    assert all_match

def test_red_sum_riscv():
    vec_len:int = 16384

    mem, cores = setup_device(Core)
    l1, l2, dst = gen_data(vec_len)

    id_l1, r_op1 = pim_device_place_data(mem, cores, l1, 0)
    id_l2, r_op2 = pim_device_place_data(mem, cores, l2, r_op1[1])
    id_dst, r_dst = pim_device_place_data(mem, cores, dst, r_op2[1])

    start_command(cores, CommandType.PIM_RED_SUM, r_op1, None, None, scalar=12)

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
    result = sum([core.get_reg(core.registers[0]) for core in cores])
    all_match: bool = result == sum(l1)
    if not all_match:
        print("output", result)
        print("expected", sum(l1))
        print(f"input1, {l1}")
        print(f"input2, {l2}")
    assert all_match

# --------------------------------
# Max Functional Correctness
# --------------------------------


def test_red_max_riscv():
    vec_len:int = 16384

    mem, cores = setup_device(Core)
    l1, l2, dst = gen_data(vec_len)

    id_l1, r_op1 = pim_device_place_data(mem, cores, l1, 0)
    id_l2, r_op2 = pim_device_place_data(mem, cores, l2, r_op1[1])
    id_dst, r_dst = pim_device_place_data(mem, cores, dst, r_op2[1])

    start_command(cores, CommandType.PIM_RED_MAX, r_op1, None, None, scalar=12)

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
    result = max([core.get_reg(core.registers[0]) for core in cores])
    all_match: bool = result == max(l1)
    if not all_match:
        print("output", result)
        print("expected", max(l1))
        print(f"input1, {l1}")
        print(f"input2, {l2}")
    assert all_match


def test_red_max_streaming():
    vec_len:int = 16384

    mem, cores = setup_device(StreamingCore)
    l1, l2, dst = gen_data(vec_len)

    id_l1, r_op1 = pim_device_place_data(mem, cores, l1, 0)
    id_l2, r_op2 = pim_device_place_data(mem, cores, l2, r_op1[1])
    id_dst, r_dst = pim_device_place_data(mem, cores, dst, r_op2[1])

    start_command(cores, CommandType.PIM_RED_MAX, r_op1, None, None, scalar=12)

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
    result = max([core.get_reg(core.registers[0]) for core in cores])
    all_match: bool = result == max(l1)
    if not all_match:
        print("output", result)
        print("expected", max(l1))
        print(f"input1, {l1}")
        print(f"input2, {l2}")
    assert all_match

# --------------------------------
# Min Functional Correctness
# --------------------------------


def test_red_min_riscv():
    vec_len:int = 16384

    mem, cores = setup_device(Core)
    l1, l2, dst = gen_data(vec_len)

    id_l1, r_op1 = pim_device_place_data(mem, cores, l1, 0)
    id_l2, r_op2 = pim_device_place_data(mem, cores, l2, r_op1[1])
    id_dst, r_dst = pim_device_place_data(mem, cores, dst, r_op2[1])

    start_command(cores, CommandType.PIM_RED_MIN, r_op1, None, None, scalar=12)

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
    result = min([core.get_reg(core.registers[0]) for core in cores])
    all_match: bool = result == min(l1)
    if not all_match:
        print("output", result)
        print("expected", min(l1))
        print(f"input1, {l1}")
        print(f"input2, {l2}")
    assert all_match


def test_red_min_streaming():
    vec_len:int = 16384

    mem, cores = setup_device(StreamingCore)
    l1, l2, dst = gen_data(vec_len)

    id_l1, r_op1 = pim_device_place_data(mem, cores, l1, 0)
    id_l2, r_op2 = pim_device_place_data(mem, cores, l2, r_op1[1])
    id_dst, r_dst = pim_device_place_data(mem, cores, dst, r_op2[1])

    start_command(cores, CommandType.PIM_RED_MIN, r_op1, None, None, scalar=12)

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
    result = min([core.get_reg(core.registers[0]) for core in cores])
    all_match: bool = result == min(l1)
    if not all_match:
        print("output", result)
        print("expected", min(l1))
        print(f"input1, {l1}")
        print(f"input2, {l2}")
    assert all_match
