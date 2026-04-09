from lib.memsys import MemSystem
from lib.dramsim import callback_t, CallbackType, dramsim3


def setup(size: int = 32):
    hbm_mem = MemSystem("./dramsim3/configs/HBM2_8Gb_x128.ini", ".", nd_log=True)
    hbm_mem.set_pim_mode(True)

    ddr4_mem = MemSystem("./dramsim3/configs/DDR4_8Gb_x16_3200.ini", ".", nd_log=True)
    ddr4_mem.set_pim_mode(True)

    test_list = list(range(size))
    return hbm_mem, ddr4_mem, test_list

def test_gdl_size():
    h, d, _ = setup()
    assert d.m_gdl_width == 128
    assert h.m_gdl_width == 512

def test_gdl_fetch_size():
    h, d, t = setup()

    def aux(mem: MemSystem, size: int):
        mem.add_data_structure(t, 4)
        mem.mmap(0, 0, 0, 1, 0, data_index=0, length=len(t) * 4, offset=0)
        assert len(mem.fetch_gdl_at(0, 0, 0, 1, 0)) == size

    aux(h, 16)
    aux(d, 4)

def test_gdl_fetch_edge_cases():
    h, d, t = setup(128)

    def aux(mem: MemSystem, size: int):
        mem.add_data_structure(t, 4)
        mem.mmap(0, 0, 0, 1, 0, data_index=0, length=len(t) * 4, offset=0)
        # not done with a for loop so we can manually inspect more about it
        assert mem.fetch_gdl_at(0, 0, 0, 1, 0) == t[0:size]
        assert mem.fetch_gdl_at(0, 0, 0, 1, 1) == t[size:2*size]
        assert mem.fetch_gdl_at(0, 0, 0, 1, 2) == t[2*size:3*size]
        assert mem.fetch_gdl_at(0, 0, 0, 1, 3) == t[3*size:4*size]
        assert mem.fetch_gdl_at(0, 0, 0, 1, 4) == t[4*size:5*size]

    aux(h, 16)
    aux(d, 4)
