from enum import Enum
from typing import Literal
from multiprocessing import Process
import numpy as np
import numpy.typing as npt


class ProcessType(Enum):
    U022 = "0.022"
    U040 = "0.040"
    U032 = "0.032"
    U090 = "0.090"


# TODO: determine how to set endianness here
class Scratchpad:
    def __init__(
        self,
        size: int = 32768,
        proc: ProcessType = ProcessType.U022,
        endianness: Literal[">", "<"] = ">",
    ):
        self.size: int = size
        self.process: ProcessType = proc
        self.access_time: np.float32 = np.float32(0.227315)
        self.data: memoryview = np.zeros(self.size, dtype=np.uint8).data
        self.endianness: Literal[">", "<"] = endianness

    def read_bytes(
        self, start: int, length: int = 1, dtype: npt.DTypeLike = np.int32
    ) -> npt.NDArray[np.generic]:
        dt = np.dtype(dtype)
        return np.frombuffer(self.data, dtype=dt.newbyteorder(self.endianness), count=length, offset=start)
