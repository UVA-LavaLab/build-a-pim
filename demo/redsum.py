import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.memsys import MemSystem
from lib.containers import Ptr
from lib.cores.bank_simd_scratch import Core
from lib.controller.commands import Command, CommandType
from lib.address.allocation import pim_device_place_data
import numpy as np
import numpy.typing as npt
import random
import time


if __name__ == "__main__":
    mem = MemSystem("./dramsim3/configs/HBM2_8Gb_x128.ini", ".")
    vec_len = int(65536 / 4) - 20
    test_list1: npt.NDArray[np.int32] = np.arange(vec_len, dtype=np.int32)
    slice_len = int(len(test_list1) / mem.num_banks)

    cores = [
        Core((c, r, bg, b), Ptr(mem))
        for c in range(mem.num_channels)
        for r in range(mem.num_ranks)
        for bg in range(mem.num_bankgroups_per_rank)
        for b in range(mem.num_banks_per_group)
    ]

    tl1id, r_op1 = pim_device_place_data(mem, cores, test_list1, 0)

    # Algorithm is reduced to a single call
    # for core in cores:
    #     core.tick(
    #         cmd=Command(CommandType.PIM_RED_MIN, 0, slice_len * 4, dtype=np.int32)
    #     )
    for core in cores:
        core.tick(
            cmd=Command(
                CommandType.PIM_RED_SUM,
                r_op1[0],
                r_op1[1],
                dtype=np.int32,
            )
        )

    start = time.perf_counter_ns()
    i = 0
    denominator = 5
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
    all_match = sum([core.rA for core in cores]) == sum(test_list1)
    print("Output Correct?", all_match)
    if not all_match:
        print("output", sum([core.rA for core in cores]))
        print("expected", sum(test_list1))
        print(f"input, {test_list1}")
    # print(f"sum: {rval} (expected {min(padded_test_list)})")
    print(f"cycles taken: {i}")
    print(f"time taken: {i * mem.tck} ns")
    print(f"real world time taken: {(stop - start) * 0.000001}")

    # sp = Scratchpad()
    # sp.data[3] = 23
    # iarr = sp.read_bytes(0, 25)
    # iarr[2] = 32
    # print(sp.read_bytes(0, 25))
