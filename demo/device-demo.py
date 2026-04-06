import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import random
from typing import Any
from lib.memsys import MemSystem


class Adder:
    def __init__(self, rank: int, bankgroup: int, bank: int):
        self.rank: int = rank
        self.bank: int = bank
        self.bankgroup: int = bankgroup
        self.is_waiting: bool = False
        self.active_addr: int = 0x0
        self.gdl: Any = [0, 0, 0, 0]
        self.regA: int = 0
        self.ptr: int = -1

    def tick(self, mem: MemSystem):
        if self.active_addr / 4 >= 4:
            return
        if self.is_waiting:
            if self.active_addr in [
                p[0] for p in mem.nd_log[0][self.rank][self.bankgroup][self.bank]
            ]:
                self.gdl = mem.fetch_gdl_at(
                    0, self.rank, self.bankgroup, self.bank, self.active_addr
                )
                mem.nd_log[0][self.rank][self.bankgroup][self.bank] = []
                self.ptr = 0
                self.is_waiting = False
        else:
            if self.ptr < 0 or self.ptr > 3:
                _ = mem.add_transaction_to_bank(
                    0,
                    self.rank,
                    self.bankgroup,
                    self.bank,
                    self.active_addr,
                    False,
                    True,
                )
                self.is_waiting = True
            else:
                self.regA += self.gdl[self.ptr]
                self.ptr += 1
                self.active_addr += 4


class Device:
    def __init__(self, config_file: str):
        self.mem: MemSystem = MemSystem(config_file, ".", nd_log=True)
        self.adders: list[Adder] = [Adder(0, 0, 0), Adder(0, 0, 1)]

    def setup(self):
        test_list_a = [1, 2, 3, 4]
        test_list_b = [5, 6, 7, 8]
        self.mem.add_data_structure(test_list_a, 4)
        self.mem.add_data_structure(test_list_b, 4)
        self.mem.mmap(
            0, 0, 0, 0, 0x0, data_index=0, length=len(test_list_a) * 4, offset=0
        )
        self.mem.mmap(
            0, 0, 0, 1, 0x0, data_index=1, length=len(test_list_b) * 4, offset=0
        )

    def tick(self):
        for a in self.adders:
            a.tick(self.mem)
        self.mem.tick()


if __name__ == "__main__":
    device = Device("./dramsim3/configs/DDR4_8Gb_x16_3200.ini")
    device.setup()
    ticks = 0
    while device.adders[0].regA < 10 and device.adders[1].regA < 26:
        ticks += 1
        device.tick()
    print("regA: core 0=", device.adders[0].regA, "core 1=", device.adders[1].regA)
    print("Cycles taken:", ticks)
