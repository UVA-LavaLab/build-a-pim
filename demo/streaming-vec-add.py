import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.memsys import MemSystem
from lib.containers import Ptr
from lib.cores.ins_stream_bank_simd import Core
from lib.controller.commands import Command, CommandType
from lib.address.allocation import pim_device_place_data
import numpy as np
import numpy.typing as npt
import random
import time


if __name__ == "__main__":
    mem = MemSystem("./dramsim3/configs/HBM2_8Gb_x128.ini", ".")
    vec_len = 1024 * 1024
    test_list1: npt.NDArray[np.int32] = np.array(
        [random.randint(0, 5) for _ in range(vec_len)], dtype=np.int32
    )
    test_list2: npt.NDArray[np.int32] = np.array(
        [random.randint(0, 5) for _ in range(vec_len)], dtype=np.int32
    )
    dst_list: npt.NDArray[np.int32] = np.array(
        [0 for _ in range(vec_len)], dtype=np.int32
    )
    slice_len = int(len(test_list1) / mem.num_banks)
    assert len(dst_list) == len(test_list1) == len(test_list2)

    cores = [
        Core((c, r, bg, b), Ptr(mem))
        for c in range(mem.num_channels)
        for r in range(mem.num_ranks)
        for bg in range(mem.num_bankgroups_per_rank)
        for b in range(mem.num_banks_per_group)
    ]

    tl1id, r_op1 = pim_device_place_data(mem, cores, test_list1, 0)
    tl2id, r_op2 = pim_device_place_data(mem, cores, test_list2, r_op1[1])
    dlid, r_dst = pim_device_place_data(mem, cores, dst_list, r_op2[1])

    # Algorithm is reduced to a single call
    # for core in cores:
    #     core.tick(
    #         cmd=Command(CommandType.PIM_RED_MIN, 0, slice_len * 4, dtype=np.int32)
    #     )
    for core in cores:
        core.tick(
            cmd=Command(
                CommandType.PIM_ADD,
                r_op1[0],
                r_op1[1],
                r_op2[0],
                r_op2[1],
                r_dst[0],
                r_dst[1],
                dtype=np.int32,
            )
        )


    start = time.perf_counter_ns()
    i = 0
    denominator = 1
    while True:
        all_done = i % denominator == 0
        if i % denominator == 0:
            for j, core in enumerate(cores):
                core.tick()
                if (
                    len(core.instruction_queue) > 0
                    or not core.pipeline.is_empty()
                    or i < 100
                ):
                    all_done = False
        mem.tick()
        i += 1
        if all_done:
            break

    stop = time.perf_counter_ns()

    # use r_dst to find outputs
    outputs_arr: npt.NDArray[np.int32] = np.frombuffer(
        mem.stored_data_structures[dlid].data_structure, dtype=np.int32
    )
    all_match = np.all(outputs_arr == (test_list1 + test_list2))
    print("All matching?", all_match)
    if not all_match:
        print("output", outputs_arr)
        print("expected", test_list1 + test_list2)
        print(f"input1, {test_list1}")
        print(f"input2, {test_list2}")
    # print(f"sum: {rval} (expected {min(padded_test_list)})")
    print(f"cycles taken: {i}")
    print(f"time taken: {i * mem.tck} ns")
    print(f"real world time taken: {(stop - start) * 0.000001}")


    # sp = Scratchpad()
    # sp.data[3] = 23
    # iarr = sp.read_bytes(0, 25)
    # iarr[2] = 32
    # print(sp.read_bytes(0, 25))
