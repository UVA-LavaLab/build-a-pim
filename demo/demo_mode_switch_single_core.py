import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.dramsim import callback_t
from lib.memsys import MemSystem
from lib.monad import DataStatus, DataWrapper, DataSetter, Ptr, DataStructureContainer
from lib.cores.mode_switcher import Core as MS
from lib.cores.components.scratchpad import Scratchpad
from lib.cores.instructions import Instruction, OpType
from lib.types import Location
from lib.controller.commands import Command, CommandType
import numpy as np
import math

if __name__ == "__main__":
    p_mem = Ptr(MemSystem("./dramsim3/configs/HBM2_8Gb_x128.ini", ".", nd_log=True))
    core = MS((0, 0, 0, 0), p_mem)
    test_list = np.arange(65536)
    _ = p_mem().add_data_structure(test_list)
    p_mem().mmap(0, 0, 0, 0, 0, 0, length=len(test_list), offset=0)

    core.add_instruction(OpType.READ, addr=0x0)
    core.add_instruction(OpType.READ, addr=0x40)
    _ = core.tick(cmd=Command(CommandType.SWITCH_MODE_MEM, location=(0, 0, 0, 0)))
    _ = core.tick(cmd=Command(CommandType.PIM_BANK_PING, location=(0, 0, 0, 0)))
    _ = core.tick(
        cmd=Command(CommandType.MEM_READ, operand_1=0x1, location=(0, 0, 0, 0))
    )
    _ = core.tick(cmd=Command(CommandType.SWITCH_MODE_PIM, location=(0, 0, 0, 0)))

    i = 0
    while True:
        all_done = True
        r = core.tick()
        if r is not None:
            print(r)
        # ensure that the pipeline is filled to some degree
        if len(core.instruction_queue) > 0 or not core.pipeline.is_empty() or i < 3:
            all_done = False
        p_mem().tick()
        i += 1
        if all_done:
            break
    print("Cycles taken:", core.cycle)
    print("Mode:", core.mode)
