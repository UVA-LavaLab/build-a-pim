from lib.dramsim import callback_t
from lib.memsys import MemSystem
from lib.monad import DataStatus, DataWrapper, DataSetter, Ptr, DataStructureContainer
from lib.cores.lobsta import Core, mkDefaultStages
from lib.cores.components.scratchpad import Scratchpad
from lib.cores.instructions import Instruction, OpType
from lib.types import Location
from lib.controller.commands import Command, CommandType
import numpy as np

if __name__ == "__main__":
    # mem = MemSystem("./dramsim3/configs/HBM2_8Gb_x128.ini", ".", nd_log=True)
    # test_list = list(range(32))

    dw = DataWrapper([24, 25, 26, 27])
    # dw = DataWrapper(np.array([24, 25, 26, 27], dtype=np.int32))
    dw.set_ready()
    print(dw)
    dw[1, np.float32] = 3
    print(dw.str_as_type(np.float32))
    print(dw[1, np.float32])

    dsc = DataStructureContainer(np.zeros(4, dtype=np.int32))
    print(dsc)
