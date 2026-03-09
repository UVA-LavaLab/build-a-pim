import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import random
from lib.memsys import MemSystem

mem = MemSystem("./dramsim3/configs/DDR4_8Gb_x16_3200.ini", ".")
mem.toggle_pim_mode()

for _ in range(4):
    mem.add_transaction_to_bank(channel=0, 
                                rank=0,
                                bankgroup=0,
                                bank=0,
                                addr=random.randint(0x0,0x100),
                                is_write=True,
                                is_pim=True)
    mem.tick(until_event=True)
    print("Transaction completed at:", mem.m_cycle,
          "Accessed addr:", hex(mem.m_writes[-1][0]))
print("The writes list:", mem.m_writes)
