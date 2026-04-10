import cProfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.memsys import MemSystem
from lib.monad import Ptr
from lib.cores.lobsta import Core
from lib.cores.instructions import OpType
import numpy as np


def main():
    mem = MemSystem("./dramsim3/configs/HBM2_8Gb_x128.ini", ".", nd_log=True)
    test_list = np.arange(65536, dtype=np.int32)

    pad_size = mem.c_num_banks - (len(test_list) % (mem.c_num_banks)) if (len(test_list) % mem.c_num_banks) == 0 else 0
    padded_test_list = np.pad(test_list, pad_width=((0, pad_size)))

    slice_len = int(len(test_list) / mem.c_num_banks)
    for i in range(int(len(padded_test_list) / slice_len)):
        mem.add_data_structure(padded_test_list[i * slice_len : (i + 1) * slice_len])

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

    for core in cores:
        core.add_instruction(OpType.READ, operands=[0x0, "reg_vA"])
        for i in range(1, int(slice_len / 16)):
            core.add_instruction(OpType.READ, operands=[0x1 * i])
            core.add_instruction(OpType.ADD, operands=["reg_vA", 0x1 * i])
        core.add_instruction(OpType.ACC, operands=["regA", "reg_vA"])

    i = 0
    while True:
        all_done = i % 5 == 0
        if i % 5 == 0:
            for j, core in enumerate(cores):
                core.tick()
                # print("core cycle:", core.cycle)
                # print("core id", j)
                # print("core ins queue:", [str(i) for i in core.instruction_queue])
                # if j == 0:
                #     print("core pipeline:", core.pipeline)
                #     print("core gdl:", core.gdl)
                #     print("core's reg_vA", core.reg_vA)
                #     print("core cycle:", core.cycle)
                #     print("----------------")
                if (
                    len(core.instruction_queue) > 0
                    or not core.pipeline.is_empty()
                    and i > 100
                ):
                    all_done = False
        mem.tick()
        i += 1
        if all_done:
            break

    rval = sum([core.regA for core in cores])
    print(f"sum: {rval} (expected {sum(padded_test_list)})")
    print(f"cycles taken: {i}")
    print(f"time taken: {i * mem.c_tck} ns")


if __name__ == "__main__":
    cProfile.run("main()")
