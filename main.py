from lib.dramsim import callback_t
from lib.memsys import MemSystem
from lib.monad import DataStatus, DataWrapper, DataSetter, Ptr
from lib.cores.lobsta import Core, mkDefaultStages
from lib.cores.instructions import Instruction, OpType
from lib.types import Location
from lib.controller.commands import Command, CommandType

if __name__ == "__main__":
    # mem = MemSystem("./dramsim3/configs/DDR4_8Gb_x16_3200.ini", ".", nd_log=True)
    mem = MemSystem("./dramsim3/configs/HBM2_8Gb_x128.ini", ".", nd_log=True)
    test_list = list(range(65536))

    padded_test_list = test_list.copy()
    for i in range(mem.c_num_banks - (len(test_list) % (mem.c_num_banks))):
        padded_test_list.append(0)

    slice_len = int(len(test_list) / mem.c_num_banks)
    for i in range(int(len(padded_test_list) / slice_len)):
        mem.add_data_structure(
            padded_test_list[i * slice_len : (i + 1) * slice_len].copy(), 4
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
                if len(core.instruction_queue) > 0 or not core.pipeline.is_empty() and i > 100:
                    all_done = False
        mem.tick()
        i += 1
        if all_done:
            break

    rval = sum([core.regA for core in cores])
    print(f"sum: {rval} (expected {sum(padded_test_list)})")
    print(f"cycles taken: {i}")
    print(f"time taken: {i * mem.c_tck} ns")
