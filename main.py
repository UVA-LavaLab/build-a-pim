from lib.dramsim import callback_t
from lib.memsys import MemSystem
from lib.monad import Ptr
from lib.cores.blimp import Core
from lib.cores.components.scratchpad import Scratchpad
from lib.cores.instructions import Instruction, OpType
from lib.types import Location
from lib.controller.commands import Command, CommandType
import numpy as np

if __name__ == "__main__":
    # mem = MemSystem("./dramsim3/configs/DDR4_8Gb_x16_3200.ini", ".", nd_log=True)
    mem = MemSystem("./dramsim3/configs/HBM2_8Gb_x128.ini", ".", nd_log=True)
    test_list = list(range(65536))

    # TODO: fix the issue with max / min and padding
    padded_test_list = test_list.copy()
    for i in range(mem.c_num_banks - (len(test_list) % (mem.c_num_banks))):
        padded_test_list.append(0)

    slice_len = int(len(test_list) / mem.c_num_banks)
    for i in range(int(len(padded_test_list) / slice_len)):
        mem.add_data_structure(
            padded_test_list[i * slice_len : (i + 1) * slice_len].copy()
        )

    for c in range(mem.c_num_channels):
        for r in range(mem.c_num_ranks):
            for bg in range(mem.c_num_bankgroups_per_rank):
                for b in range(mem.c_num_banks_per_group):
                    mem.mmap(
                        c,
                        r,
                        bg,
                        b,
                        0,
                        data_index=c
                        * mem.c_num_ranks
                        * mem.c_num_bankgroups_per_rank
                        * mem.c_num_banks_per_group
                        + r * mem.c_num_bankgroups_per_rank * mem.c_num_banks_per_group
                        + bg * mem.c_num_banks_per_group
                        + b,
                        length=slice_len * 4,
                        offset=0,
                    )

    cores = [
        Core((c, r, bg, b), Ptr(mem))
        for c in range(mem.c_num_channels)
        for r in range(mem.c_num_ranks)
        for bg in range(mem.c_num_bankgroups_per_rank)
        for b in range(mem.c_num_banks_per_group)
    ]

    # Algorithm is reduced to a single call
    for core in cores:
        core.tick(cmd=Command(CommandType.PIM_RED_MIN, 0, slice_len * 4, dtype=np.int32))

    i = 0
    denominator = 5
    while True:
        all_done = i % denominator == 0
        if i % denominator == 0:
            for j, core in enumerate(cores):
                core.tick()
                if j == 0:
                    print("----------------")
                    # print("core pipeline:", core.pipeline)
                    for s in core.pipeline.stages:
                        print(s.name, ":", s.ins)
                    print("core cycle:", core.cycle)
                    # print("mem cycle:", mem.m_cycle)
                    # print("core gdl:", core.gdl)
                    # print("core's reg_vA", core.reg_vA)
                if len(core.instruction_queue) > 0 or not core.pipeline.is_empty() and i > 100:
                    all_done = False
        mem.tick()
        i += 1
        if all_done:
            break

    rval = min([core.regA for core in cores])
    print(f"sum: {rval} (expected {min(padded_test_list)})")
    print(f"vector register in core 0:", cores[0].reg_vA)
    print(f"cycles taken: {i}")
    print(f"time taken: {i * mem.c_tck} ns")

    sp = Scratchpad()
    sp.data[3] = 23
    iarr = sp.read_bytes(0, 25)
    iarr[2] = 32
    print(sp.read_bytes(0, 25))


