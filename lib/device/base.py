from collections import deque
from lib.address.allocation import pim_device_place_data
from lib.controller.commands import Command
from lib.cores.components.base import BaseCore
from lib.memsys import MemSystem
from lib.monad import Ptr
from lib.controller.controller import (
    Controller,
    ControllerState,
    BaselineState,
    Transaction,
)
from lib.controller.response import Response
from lib.errors import PimCrammedResponseError

import sys
import numpy as np
import numpy.typing as npt


def crammed[T: BaseCore](dev: BaseDevice[T], bits: int):
    raise PimCrammedResponseError(
        f"Cumulative bits sent along the GDL {bits} is larger than device GDL"
        + f" width {dev.mem.m_gdl_width} during cycle {dev.cycle}"
    )


class BaseDevice[T: BaseCore]:
    def __init__(self, core_type: type[T], config: str, cores: list[T] | None = None, trans_queue_len: int = sys.maxsize):
        """
        The device class creates its own managed memory system, cores, and PIM
        memory controller. It is the intended interface point for a simulated
        PIM device.
        """
        # TODO: incorporate logic for different-frequency components
        self.cycle: int = 0
        self.mem: MemSystem = MemSystem(config, ".", nd_log=True)
        p_mem: Ptr[MemSystem] = Ptr(self.mem)

        # for clarity
        n_bank = p_mem().c_num_banks_per_group
        n_bankgroup = p_mem().c_num_bankgroups_per_rank
        n_rank = p_mem().c_num_ranks
        n_channel = p_mem().c_num_channels

        self.trans_queue_len: int = trans_queue_len

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

        self.controller: Controller[BaselineState] = Controller.baseline(p_mem, config, threshold=3)
        def emit_if_nonempty(b: ControllerState[BaselineState]) -> Command | None:
            if len(b._command_queue) > 0:
                return b._command_queue[0]
        self.controller.cmd_functions = [emit_if_nonempty]
        self._last_allocated_location: int = 0

        self._transaction_queue: deque[Transaction] = deque()

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
        self._last_allocated_location = addr_range[1]
        self.controller.malloc_obj(id, addr_range[1] - addr_range[0], addr_range[0])

        return id

    def add_transaction(self, trans: Transaction) -> bool:
        """
        Attempts to append the transaction to the transaction queue. If the
        transaction would overflow the queue length, it is not added.
        """
        if len(self._transaction_queue) >= self.trans_queue_len:
            return False
        else:
            self._transaction_queue.append(trans)
            return True

    def tick(self):
        """
        Progress the state of the device by one device cycle. Depending on the
        relative clock rate of its cores, this may or may not progress the
        state of the device's cores.
        """
        bits: int = 0
        if len(self._transaction_queue) > 0:
            self.controller.push_transaction(self._transaction_queue.popleft())
        cmd = self.controller.tick()

        responses: list[Response] = []
        for core in self.cores:
            r = core.tick(cmd)
            if r is not None:
                responses.append(r)
                bits += r.bits

        if bits > self.mem.m_gdl_width:
            crammed(self, bits)

        for r in responses:
            self.controller.push_response(r)
        self.mem.tick()
        self.cycle += 1

    def all_cores_idle(self) -> bool:
        # FIXME: this also does not type check, but works because this is python...
        # note: base core class cannot contain a pipeline since this would
        # cause a cyclic dependency
        return all(core.pipeline.is_empty() for core in self.cores)

    def transaction_queue_empty(self) -> bool:
        return len(self._transaction_queue) == 0
