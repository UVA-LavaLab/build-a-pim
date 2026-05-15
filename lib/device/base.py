from collections import deque
from lib.address.allocation import pim_device_place_data
from lib.cores.components.base import BaseCore
from lib.device.device_commands import DeviceCommand
from lib.memsys import MemSystem
from lib.containers import Ptr
from lib.errors import PimCrammedResponseError

import sys
import numpy as np
import numpy.typing as npt


def _crammed[T: BaseCore](dev: BaseDevice[T], bits: int):
    raise PimCrammedResponseError(
        f"Cumulative bits sent along the GDL {bits} is larger than device GDL"
        + f" width {dev.mem.m_gdl_width} during cycle {dev.cycle}"
    )


class BaseDevice[T: BaseCore]:
    """
    A device class which can be built upon for more specific use cases. By default, it advances the state of 
    """
    def __init__(
        self,
        core_type: type[T],
        config: str,
        cores: list[T] | None = None,
        core_relative_clock_rate: int = 1,
        trans_queue_len: int = sys.maxsize,
    ):
        """
        The device class creates its own managed memory system, cores, and PIM
        memory controller. It is the intended interface point for a simulated
        PIM device.

        The core relative clock rate determines how much slower the associated
        core is relative to the memory system. For now, this is limited to
        integer values.
        """
        # TODO: incorporate logic for different-frequency components
        self.cycle: int = 0
        self.mem: MemSystem = MemSystem(config, ".", nd_log=True)
        p_mem: Ptr[MemSystem] = Ptr(self.mem)

        # for clarity
        n_bank = p_mem().num_banks_per_group
        n_bankgroup = p_mem().num_bankgroups_per_rank
        n_rank = p_mem().num_ranks
        n_channel = p_mem().num_channels

        self.trans_queue_len: int = trans_queue_len
        self.relative_core_rate: int = core_relative_clock_rate

        # create a list of cores if none are provided (useful for different core mappings)
        self.cores: list[T] = (
            [
                core_type((c, r, bg, b), p_mem)
                for c in range(n_channel)
                for r in range(n_rank)
                for bg in range(n_bankgroup)
                for b in range(n_bank)
            ]
            if cores is None
            else cores
        )

        self._last_allocated_location: int = 0
        self._transaction_queue: deque[DeviceCommand] = deque()
        self._id_mapping: dict[int, tuple[int, int]] = {}

    def instant_place_data(
        self, arr: npt.NDArray[np.generic], local_addr: int = -1
    ) -> int:
        """
        Accepts an array which should be mapped to the device and returns a PIM
        object ID in the form of an int after mapping the object to the device.

        The optional local_addr parameter is the specific location at which the
        object should be mapped in each core's local address space. Default
        behavior will place the next object immediately after the previously
        mapped object.

        IMPORTANT: This function is meant for testing purposes and performs an
        instantaneous data transfer from host to device. If timing data
        transfer is important, do NOT use this function.
        """
        # FIXME: this doesn't type check but works in Python, so...
        addr = self._last_allocated_location if local_addr <= -1 else local_addr

        id, addr_range = pim_device_place_data(self.mem, self.cores, arr, addr)
        self._id_mapping[id] = addr_range
        self._last_allocated_location = addr_range[1]

        return id

    def add_transaction(self, cmd: DeviceCommand) -> bool:
        """
        Attempts to append the transaction to the transaction queue. If the
        transaction would overflow the queue length, it is not added.
        """
        if len(self._transaction_queue) >= self.trans_queue_len:
            return False
        else:
            self._transaction_queue.append(cmd)
            return True

    def tick(self):
        """
        Progress the state of the device by one device cycle. Depending on the
        relative clock rate of its cores, this may or may not progress the
        state of the device's cores.
        """
        bits: int = 0
        cmd = (
            self._transaction_queue.popleft().to_command(self._id_mapping)
            if len(self._transaction_queue) > 0
            else None
        )

        if self.cycle % self.relative_core_rate == 0:
            for core in self.cores:
                r = core.tick(cmd)
                if r is not None:
                    bits += r.bits

        # supposing there is some hardware support for multiple banks sending
        # data along the GDL concurrently, this is designed to prevent overflow
        # of the GDL
        if bits > self.mem.m_gdl_width:
            _crammed(self, bits)

        self.mem.tick()
        self.cycle += 1

    def all_cores_idle(self) -> bool:
        # FIXME: this also does not type check, but works because this is python...
        # NOTE: base core class cannot contain a pipeline since this would
        # cause a cyclic dependency
        return all(core.pipeline.is_empty() for core in self.cores)

    def transaction_queue_empty(self) -> bool:
        return len(self._transaction_queue) == 0
