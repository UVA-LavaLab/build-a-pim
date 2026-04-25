from lib.errors import PimCrammedResponseError
from lib.memsys import MemSystem
from lib.monad import Ptr

class Response:
    def __init__(self, p_mem: Ptr[MemSystem], response_bits: int, active_row: int | None = None):
        if response_bits > p_mem().m_gdl_width:
            raise PimCrammedResponseError("Response over-crowded with data (cannot possibly fit on GDL)")
        self.active_row: int = active_row if active_row is not None else -1
