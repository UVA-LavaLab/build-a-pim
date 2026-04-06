import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import random
from typing import Any
from lib.memsys import MemSystem
from lib.monad import DataSetter

if __name__ == "__main__":
    mem = MemSystem("./dramsim3/configs/DDR4_8Gb_x16_3200.ini", ".", nd_log=True)
    test_list = list(range(32))
    mem.add_data_structure(test_list, 4)
    mem.mmap(0, 0, 0, 0, 0, data_index=0, length=len(test_list) * 4, offset=0)

    gdl = mem.get((0, 0, 0, 0, 0))
    print("initial gdl value", gdl)
    mem.tick(until_event=True)
    while not gdl.is_ready():
        mem.tick(until_event=True)
    gdl.data[1] = 55
    print("gdl after data modification", gdl)
    new_gdl = mem.set((0, 0, 0, 0, 0), gdl)
    mem.tick(until_event=True)
    while not new_gdl.is_ready():
        mem.tick(until_event=True)
    print("updated data wrapper", new_gdl)
    gdl = mem.get((0, 0, 0, 0, 0))
    mem.tick(until_event=True)
    while not gdl.is_ready():
        mem.tick(until_event=True)
    print("gdl after reloading stored data", gdl)
