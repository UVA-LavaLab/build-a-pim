from lib.cores.components.pipeline import Pipeline
from lib.cores.instructions import IState as IS, OpType as OT
from lib.memsys import MemSystem
from lib.monad import Ptr, DataWrapper
from lib.cores.bank_simd import Core as SC
import numpy as np
import numpy.typing as npt
import math


def dev_setup() -> tuple[MemSystem, SC, Pipeline]:
    mem = MemSystem("./dramsim3/configs/HBM2_8Gb_x128.ini", ".", nd_log=True)
    core = SC((0, 0, 0, 0), Ptr(mem))
    return (mem, core, core.pipeline)


def list_setup() -> (
    tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32]]
):
    base = np.zeros(128, dtype=np.int32) + 1
    asc = np.arange(128, dtype=np.int32)
    desc = np.arange(0, 128, step=-1, dtype=np.int32)

    return (base, asc, desc)


def map_list_at_0(mem: MemSystem, base: npt.NDArray[np.int32]):
    id = mem.add_data_structure(base)

    base_bytes = np.frombuffer(base, dtype=np.uint8)
    gdl_bytes: int = int(mem.m_gdl_width / 8)
    len_chunks: int = int(math.ceil(len(base_bytes) / gdl_bytes))

    mem.mmap(0, 0, 0, 0, 0, id, len_chunks, offset=0)

def test_core_read_same_location_consecutively():
    mem, core, pipe = dev_setup()
    base, _, _ = list_setup()

    map_list_at_0(mem, base)
    core.add_instruction(OT.READ, addr=0x0, dst="reg_vA")
    core.add_instruction(OT.READ, addr=0x0, dst="reg_vA")
    counter = 0
    while not pipe.is_empty() and counter < 100:
        core.tick()
        mem.tick()
        counter += 1

    assert counter < 100


def test_core_write_same_location_consecutively():
    mem, core, pipe = dev_setup()
    base, asc, _ = list_setup()

    map_list_at_0(mem, base)

    dw = DataWrapper(np.frombuffer(asc, count=int(mem.m_gdl_width / 32)))
    core.set_reg("reg_vA", dw)

    core.add_instruction(OT.WRITE, addr=0x0, in_reg1="reg_vA")
    core.add_instruction(OT.WRITE, addr=0x0, in_reg1="reg_vA")
    counter = 0
    while not pipe.is_empty() and counter < 100:
        core.tick()
        mem.tick()
        counter += 1

    assert counter < 100
