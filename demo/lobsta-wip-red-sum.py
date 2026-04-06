import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.memsys import MemSystem
from lib.monad import Ptr
from lib.cores.lobsta import Core
from lib.cores.instructions import OpType

if __name__ == "__main__":
    mem = MemSystem("./dramsim3/configs/DDR4_8Gb_x16_3200.ini", ".", nd_log=True)
    test_list = list(range(128))
    mem.add_data_structure(test_list, 4)
    mem.mmap(0, 0, 0, 0, 0, data_index=0, length=len(test_list) * 4, offset=0)

    core = Core((0, 0, 0, 0), Ptr(mem))
    core.add_instruction(OpType.NOP)
    core.add_instruction(OpType.READ, operands=[0x0, "reg_vA"])
    for i in range(1, int(len(test_list) / 4)):
        core.add_instruction(OpType.READ, operands=[0x10 * i])
        core.add_instruction(OpType.ADD, operands=["reg_vA", 0x10 * i])

    core.add_instruction(OpType.ACC, operands=["regA", "reg_vA"])

    while len(core.instruction_queue) > 0 or not core.pipeline.is_empty():
        core.tick()
        mem.tick()

    print("test list (first 16/128):", test_list[0:16])
    print("reg_vA:", core.reg_vA)
    print("regA:", core.regA, f"(expected {sum(test_list)})")
    print("cycles taken:", core.cycle)
    print("time:", core.cycle * mem.c_tck, "ns")
