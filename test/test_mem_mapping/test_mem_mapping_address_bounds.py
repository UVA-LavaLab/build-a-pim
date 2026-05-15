from lib.memsys import MemSystem
from lib.dramsim import callback_t, CallbackType, dramsim3
import numpy as np
import numpy.typing as npt
import math


def setup(size: int = 32):
    hbm_mem = MemSystem("./dramsim3/configs/HBM2_8Gb_x128.ini", ".", nd_log=True)
    hbm_mem.set_pim_mode(True)

    ddr4_mem = MemSystem("./dramsim3/configs/DDR4_8Gb_x16_3200.ini", ".", nd_log=True)
    ddr4_mem.set_pim_mode(True)

    test_list = np.arange(size)
    return hbm_mem, ddr4_mem, test_list


def test_mmap_row_boundaries_invertible():
    hbm_mem, ddr4_mem, test_list = setup(128)

    def t(mem: MemSystem, dtype: npt.DTypeLike = np.int32):
        dtype = np.dtype(dtype)
        gdl_width_words = int(math.ceil(mem.m_gdl_width / (8 * 4)))
        gdl_chunk_1 = test_list[:gdl_width_words]
        gdl_chunk_2 = test_list[gdl_width_words : gdl_width_words * 2]
        num_col = mem.get_config_param("n_col")

        _ = mem.add_data_structure(np.array(gdl_chunk_1, dtype=dtype))
        _ = mem.add_data_structure(np.array(gdl_chunk_2, dtype=dtype))
        mem.mmap(0, 0, 0, 0, 0, data_index=0, length=len(gdl_chunk_1) * 4, offset=0)
        mem.mmap(
            0, 0, 0, 0, num_col, data_index=1, length=len(gdl_chunk_2) * 4, offset=0
        )

        chunk_1 = mem.get((0, 0, 0, 0, 0), dtype=dtype)
        chunk_2 = mem.get((0, 0, 0, 0, num_col), dtype=dtype)

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
