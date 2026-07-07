from lib.containers import Box, Ptr
from lib.cores.components.addressing import Swizzler
from lib.memsys import MemSystem
from typing import Any


# the idea of this class is that it manages all of the data readiness stuff
# we can move all of the instruction-level checks for data readiness into this
class DataBuffer:
    def __init__(
        self,
        p_mem: Ptr[MemSystem],
        locations: list[tuple[int, int, int, int]] | None = None,
        swizzler: Swizzler | None = None,
    ):
        """
        Accepts a pointer to a memory system which it checks for readiness.
        Also accepts a list of locations which it should poll for returned data.
        """
        self.contracts: list[Box] = []
        self.p_mem: Ptr[MemSystem] = p_mem
        self.swizzled: bool = swizzler is not None
        self.swizzler: Swizzler = Swizzler([]) if swizzler is None else swizzler

    # check all of the open contracts to see if they are ready
    # if they are ready, then remove them from the open contract list
    def tick(self):
        def rev_enum(data: list[Box]):
            for i in range(len(data) - 1, -1, -1):
                yield (i, data[i])

        for i, contract in rev_enum(self.contracts):
            if contract.is_ready():
                # remove from list
                del self.contracts[i]

    # returns a Box which contains the requested data which will be readied when appropriate
    def get(self, location: tuple[int, int, int, int], addr: int, swizzle: bool = True):
        # TODO: make this properly return data
        dev_addr = self.p_mem().loc_to_device_addr(*location, addr)
        if self.swizzled and swizzle:
            dev_addr = self.swizzler(dev_addr)

        contract = self.p_mem().get(dev_addr)
        self.contracts.append(contract)
        return contract

    def set(
        self,
        location: tuple[int, int, int, int],
        addr: int,
        data: Box,
        swizzle: bool = True,
    ):
        # TODO: correct this to set the data using the memory pointer
        dev_addr = self.p_mem().loc_to_device_addr(*location, addr)
        if self.swizzled and swizzle:
            dev_addr = self.swizzler(dev_addr)

        contract = self.p_mem().set(dev_addr, data)
        self.contracts.append(contract)
        return contract
