import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.cores.components import scratchpad
from lib.dramsim import callback_t
from lib.memsys import MemSystem
from lib.monad import Ptr
from lib.cores.bank_simd_scratch import Core
from lib.cores.components.scratchpad import Scratchpad
from lib.cores.instructions import Instruction, OpType
from lib.types import Location
from lib.controller.commands import Command, CommandType
from lib.address.allocation import pim_device_place_data
import numpy as np
import numpy.typing as npt
import random
import math


if __name__ == "__main__":
    # create a memory system, core, and demo list
    mem = MemSystem("./dramsim3/configs/HBM2_8Gb_x128.ini", ".", nd_log=True)
    core = Core((0, 0, 0, 0), Ptr(mem))
    vec_len = 128
    test_list1: npt.NDArray[np.int32] = np.array(
        [random.randint(1, 2) for _ in range(vec_len)], dtype=np.int32
    )

    # map the data onto the device
    tl1id, r_op1 = pim_device_place_data(mem, [core], test_list1, 0)

    # add some instructions
    core.add_instruction(OpType.READ, dst="reg_vA", addr=0x0)
    core.add_instruction(OpType.SCRATCH_WRITE, in_reg1="reg_vA", addr=0x0)
    core.add_instruction(OpType.SCRATCH_READ, dst="reg_vB", addr=0x0)

    # tick for some length longer than the runtime of instructions
    for _ in range(50):
        core.tick()
        mem.tick()

    print(core.instruction_queue)
    print(core.pipeline)
    print(np.frombuffer(core.scratchpad.data[0:64], dtype=np.int32))
    print(core.reg_vB)
