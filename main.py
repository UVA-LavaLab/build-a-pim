from lib.dramsim import callback_t
from lib.memsys import MemSystem
from lib.monad import Ptr
from lib.cores.blimp import Core
from lib.cores.components.scratchpad import Scratchpad
from lib.cores.instructions import Instruction, OpType
from lib.types import Location
from lib.controller.commands import Command, CommandType
import numpy as np
import numpy.typing as npt
import random


if __name__ == "__main__":
    # mem = MemSystem("./dramsim3/configs/DDR4_8Gb_x16_3200.ini", ".", nd_log=True)
    mem = MemSystem("./dramsim3/configs/HBM2_8Gb_x128.ini", ".", nd_log=True)
    vec_len = 65536
    test_list1: npt.NDArray[np.int32] = np.array(
        [random.randint(0, 5) for _ in range(vec_len)], dtype=np.int32
    )
    test_list2: npt.NDArray[np.int32] = np.array(
        [random.randint(0, 5) for _ in range(vec_len)], dtype=np.int32
    )
    dst_list: npt.NDArray[np.int32] = np.array(
        [0 for _ in range(vec_len)], dtype=np.int32
    )
    slice_len = int(len(test_list1) / mem.c_num_banks)
    assert len(dst_list) == len(test_list1) == len(test_list2)

    # TODO: fix the issue with max / min and padding
    def map_list(L: npt.NDArray[np.int32], start: int = 0) -> tuple[int, int]:
        data_ind = mem.get_num_data_structures()
        pl = np.copy(L)
        pl = np.pad(pl, pad_width=(0, mem.c_num_banks - (len(L) % (mem.c_num_banks))))
        # for i in range(mem.c_num_banks - (len(L) % (mem.c_num_banks))):
        #     pl.append(0)
        slice_len = int(len(L) / mem.c_num_banks)
        for i in range(int(len(pl) / slice_len)):
            _ = mem.add_data_structure(pl[i * slice_len : (i + 1) * slice_len].copy())

        for c in range(mem.c_num_channels):
            for r in range(mem.c_num_ranks):
                for bg in range(mem.c_num_bankgroups_per_rank):
                    for b in range(mem.c_num_banks_per_group):
                        idx = (
                            c
                            * mem.c_num_ranks
                            * mem.c_num_bankgroups_per_rank
                            * mem.c_num_banks_per_group
                            + r
                            * mem.c_num_bankgroups_per_rank
                            * mem.c_num_banks_per_group
                            + bg * mem.c_num_banks_per_group
                            + b
                            + data_ind
                        )
                        # print("memory mapping", mem.stored_data_structures[idx], idx)
                        mem.mmap(
                            c,
                            r,
                            bg,
                            b,
                            start,
                            data_index=idx,
                            # TODO: calculate length programmatically based on data type
                            length=slice_len,
                            offset=0,
                        )
        return (start, start + slice_len)

    r_op1 = map_list(test_list1)
    r_op2 = map_list(test_list2, start=r_op1[1])
    r_dst = map_list(dst_list, start=r_op2[1])

    cores = [
        Core((c, r, bg, b), Ptr(mem))
        for c in range(mem.c_num_channels)
        for r in range(mem.c_num_ranks)
        for bg in range(mem.c_num_bankgroups_per_rank)
        for b in range(mem.c_num_banks_per_group)
    ]

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

    i = 0
    denominator = 5
    while True:
        all_done = i % denominator == 0
        if i % denominator == 0:
            for j, core in enumerate(cores):
                core.tick()
                # if j == 0:
                    # print("----------------")
                    # print("core pipeline:", core.pipeline)
                    # for s in core.pipeline.stages:
                    #     print(
                    #         s.name,
                    #         ":",
                    #         s.ins,
                    #         s.ins.ret() if s.ins is not None else "NONE",
                    #     )
                    # print("core cycle:", core.cycle)
                    # print(core.gdl)
                    # print("mem cycle:", mem.m_cycle)
                    # print("core gdl:", core.gdl)
                    # print("core's reg_vA", core.reg_vA)
                if (
                    len(core.instruction_queue) > 0
                    or not core.pipeline.is_empty()
                    and i > 100
                ):
                    all_done = False
        mem.tick()
        i += 1
        if all_done or mem.m_cycle > 120:
            break


    # use r_dst to find outputs
    outputs: list[npt.NDArray[np.int32]] = []
    for i, _ in enumerate(cores):
        outputs.append(np.frombuffer(mem.stored_data_structures[i + 256].data_structure, dtype=np.int32))
    outputs_arr: npt.NDArray[np.int32] = np.array(outputs)
    outputs_arr = outputs_arr.flatten()
    all_match = np.all(outputs_arr == (test_list1 + test_list2))
    print("All matching?", all_match)
    if not all_match:
        print("output", outputs_arr)
        print("expected", test_list1 + test_list2)
        print(f"input1, {test_list1}")
        print(f"input2, {test_list2}")
    # print(f"sum: {rval} (expected {min(padded_test_list)})")
    print(f"vector register in core 0:", cores[0].reg_vA)
    print(f"cycles taken: {i}")
    print(f"time taken: {i * mem.c_tck} ns")

    # sp = Scratchpad()
    # sp.data[3] = 23
    # iarr = sp.read_bytes(0, 25)
    # iarr[2] = 32
    # print(sp.read_bytes(0, 25))
