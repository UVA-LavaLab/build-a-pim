from lib.dramsim import callback_t
from lib.memsys import MemSystem
from lib.monad import DataStatus, DataWrapper, DataSetter, Ptr, DataStructureContainer
from lib.cores.mode_switcher import Core as MS
from lib.cores.components.scratchpad import Scratchpad
from lib.cores.instructions import Instruction, OpType
from lib.types import Location
from lib.controller.commands import Command, CommandType
import numpy as np
import math


def setup(size: int = 32):
    hbm_mem = MemSystem("./dramsim3/configs/HBM2_8Gb_x128.ini", ".", nd_log=True)
    hbm_mem.set_pim_mode(True)

    ddr4_mem = MemSystem("./dramsim3/configs/DDR4_8Gb_x16_3200.ini", ".", nd_log=True)
    ddr4_mem.set_pim_mode(True)

    test_list = list(range(size))
    return hbm_mem, ddr4_mem, test_list


def test_data_in_wrapper_4_byte():
    hbm_mem, ddr4_mem, test_list = setup()

    def t(mem: MemSystem):
        mem.add_data_structure(test_list)
        mem.mmap(0, 0, 0, 1, 0, data_index=0, length=len(test_list) * 4, offset=0)
        rval = mem.get((0, 0, 0, 1, 0), np.int32)
        print(rval.str_as_type(np.int32))
        assert np.all(rval.data == np.arange(int(mem.m_gdl_width / 32), dtype=np.int32))

    t(hbm_mem)
    t(ddr4_mem)


def test_data_in_wrapper_8_byte():
    hbm_mem, ddr4_mem, test_list = setup(128)

    def t(mem: MemSystem):
        mem.add_data_structure(np.array(test_list, dtype=np.int64))
        mem.mmap(0, 0, 0, 1, 0, data_index=0, length=len(test_list) * 8, offset=0)
        assert np.all(
            mem.get((0, 0, 0, 1, 0), dtype=np.int64).data
            == np.arange(int(mem.m_gdl_width / 64), dtype=np.int64)
        )
        assert np.all(
            mem.get((0, 0, 0, 1, 1), dtype=np.int64).data
            == np.arange(
                int(mem.m_gdl_width / 64), 2 * int(mem.m_gdl_width / 64), dtype=np.int64
            )
        )

        assert np.all(
            mem.get((0, 0, 0, 1, 2), dtype=np.int64).data
            == np.arange(
                int(mem.m_gdl_width / 64) * 2,
                int(mem.m_gdl_width / 64) * 3,
                dtype=np.int64,
            )
        )

    t(hbm_mem)
    t(ddr4_mem)


def setup_2(size: int = 32):
    hbm_mem = MemSystem("./dramsim3/configs/HBM2_8Gb_x128.ini", ".", nd_log=True)
    hbm_mem.set_pim_mode(True)

    ddr4_mem = MemSystem("./dramsim3/configs/DDR4_8Gb_x16_3200.ini", ".", nd_log=True)
    ddr4_mem.set_pim_mode(True)

    test_list = np.arange(size)
    return hbm_mem, ddr4_mem, test_list


def test_mmap_row_boundaries_invertible():
    hbm_mem, ddr4_mem, test_list = setup_2(128)

    def t(mem: MemSystem, dtype: npt.DTypeLike = np.int32):
        dtype = np.dtype(dtype)
        gdl_width_words = int(math.ceil(mem.m_gdl_width / (8 * 4)))
        gdl_chunk_1 = test_list[:gdl_width_words]
        gdl_chunk_2 = test_list[gdl_width_words : gdl_width_words * 2]
        n_col = mem.get_config_param("n_col")

        _ = mem.add_data_structure(np.array(gdl_chunk_1, dtype=dtype))
        _ = mem.add_data_structure(np.array(gdl_chunk_2, dtype=dtype))
        mem.mmap(0, 0, 0, 0, 0, data_index=0, length=len(gdl_chunk_1) * 4, offset=0)
        mem.mmap(0, 0, 0, 0, n_col, data_index=1, length=len(gdl_chunk_2) * 4, offset=0)

        chunk_1 = mem.get((0, 0, 0, 0, 0), dtype=dtype)
        chunk_2 = mem.get((0, 0, 0, 0, n_col), dtype=dtype)

        rc_1 = np.frombuffer(chunk_1.data, dtype=dtype)
        rc_2 = np.frombuffer(chunk_2.data, dtype=dtype)

        print("chunk 1:", gdl_chunk_1, "chunk recieved 1:", rc_1)
        print("chunk 2:", gdl_chunk_2, "chunk recieved 2:", rc_2)

        print(rc_1 == gdl_chunk_1)
        print(rc_2 == gdl_chunk_2)

        assert np.all(rc_1 == gdl_chunk_1)
        assert np.all(rc_2 == gdl_chunk_2)

    t(hbm_mem)
    t(ddr4_mem)


if __name__ == "__main__":
    p_mem = Ptr(MemSystem("./dramsim3/configs/HBM2_8Gb_x128.ini", ".", nd_log=True))
    core = MS((0, 0, 0, 0), p_mem)
    test_list = np.zeros(shape=(65536))
    _ = p_mem().add_data_structure(test_list)
    p_mem().mmap(0, 0, 0, 0, 0, 0, length=len(test_list), offset=0)

    core.add_instruction(OpType.READ, addr=0x0)
    core.add_instruction(OpType.READ, addr=0x40)

    i = 0
    while True:
        all_done = True
        r = core.tick(cmd=Command(0, CommandType.PIM_BANK_PING, location=(0, 0, 0, 0)))
        if r is not None:
            print(r)
        # ensure that the pipeline is filled to some degree
        if len(core.instruction_queue) > 0 or not core.pipeline.is_empty() or i < 3:
            all_done = False
        p_mem().tick()
        i += 1
        if all_done:
            break
    print("Cycles taken:", i)
