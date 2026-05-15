from enum import Enum
from typing import Literal
import numpy as np
import numpy.typing as npt
from lib.containers import Box
from lib.cores.components.base import BaseCore
from math import ceil


class ProcessType(Enum):
    """
    This currently does nothing, but will later affect the area and
    access time of the scratchpad once CACTI has been more fully
    integrated.
    """
    U022 = "0.022"
    U040 = "0.040"
    U032 = "0.032"
    U090 = "0.090"


# TODO: determine how to properly model endianness here
class Scratchpad:
    """
    A class which models the behavior of a Scratchpad module.
    """
    def __init__(
        self,
        size: int = 32768,
        cycle_time: float = 0.227315,
        bus_width: int = 512,
        proc: ProcessType = ProcessType.U022,
        # endianness: Literal[">", "<"] = ">",
    ):
        self.size: int = size
        self.process: ProcessType = proc
        self.access_time: np.float32 = np.float32(cycle_time)
        self.output_bus_width: np.int32 = np.int32(bus_width)
        # initialize an empty array of the specified size because scratchpads
        # are usually 64 KB, the overhead of this strategy is ~8 MB. even for
        # extremely large scratchpads (ex: 256 KB), the overhead will be ~32
        # MB, which is orders of magnitude smaller than bank memory simulation
        self.data: memoryview = np.zeros(self.size, dtype=np.uint8).data
        # TODO: support different endiannesses at low overhead
        # self.endianness: Literal[">", "<"] = endianness

    def get_relative_cycle_length(self, core: BaseCore) -> np.float32:
        """
        Returns the number of cycles required to access the scratchpad (for
        both read and write).
        """
        return np.float32(np.ceil(self.access_time / core.tCK))

    def read_bytes(self, core: BaseCore, addr: int) -> Box:
        """
        Reads are executed via output-bus-width addressing units, meaning for
        an output bus width of 512 bits, 0x1 fetches the first 512 bits of the
        scratchpad and 0x2 fetches the second 512 bits.

        This object is stateful, so do not use data before calling is_ready()
        on the returned Box.
        """
        deadline = core.cycle + self.get_relative_cycle_length(core)

        dw = Box(
            data=np.array([]),
        )

        def u():
            c: bool = bool(core.cycle > deadline)
            if c:
                dw.data = np.copy(
                    np.frombuffer(
                        self.data,
                        count=np.int32(self.output_bus_width / 8),
                        offset=addr * np.int32(self.output_bus_width / 8),
                        dtype=np.uint8,
                    )
                ).data
            return c

        dw.update_func = u

        return dw

    def write_bytes(
        self,
        core: BaseCore,
        addr: int,
        data: npt.NDArray[np.generic],
    ) -> Box:
        """
        Returns a Box. Data will only be stored on the cycle of termination,
        so is_ready() must be called on the returned object.
        """
        deadline = core.cycle + self.get_relative_cycle_length(core)
        dst = np.frombuffer(
            self.data,
            count=np.int32(self.output_bus_width / 8),
            offset=addr * np.int32(self.output_bus_width / 8),
            dtype=np.uint8,
        )

        def u() -> bool:
            c: bool = bool(core.cycle > deadline)
            if c:
                values = np.frombuffer(data.data, dtype=np.uint8)
                dst[0 : len(values)] = values
            return c

        return Box(
            data=np.frombuffer(data.data),
            update_func=u,
        )
