from lib.memsys import MemSystem
from lib.dramsim import callback_t, CallbackType, dramsim3


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
        mem.add_data_structure(test_list, 4)
        mem.mmap(0, 0, 0, 1, 0, data_index=0, length=len(test_list) * 4, offset=0)
        print(mem[0, 0, 0, 1, 0])
        assert mem[0, 0, 0, 1, 0].data == list(range(int(mem.m_gdl_width / 32)))

    t(hbm_mem)
    t(ddr4_mem)


def test_data_in_wrapper_8_byte():
    hbm_mem, ddr4_mem, test_list = setup(128)

    def t(mem: MemSystem):
        mem.add_data_structure(test_list, 8)
        mem.mmap(0, 0, 0, 1, 0, data_index=0, length=len(test_list) * 8, offset=0)
        assert mem[0, 0, 0, 1, 0].data == list(range(int(mem.m_gdl_width / 64)))
        assert mem[0, 0, 0, 1, 1].data == list(
            range(int(mem.m_gdl_width / 64), 2 * int(mem.m_gdl_width / 64))
        )
        assert mem[0, 0, 0, 1, 2].data == list(
            range(int(mem.m_gdl_width / 64) * 2, int(mem.m_gdl_width / 64) * 3)
        )

    t(hbm_mem)
    t(ddr4_mem)


def test_data_ready_timing():
    hbm_mem, ddr4_mem, test_list = setup()

    def t(mem: MemSystem):
        mem.add_data_structure(test_list, 4)
        mem.mmap(0, 0, 0, 1, 0, data_index=0, length=len(test_list) * 4, offset=0)
        chunk = mem[0, 0, 0, 1, 0]
        assert not chunk.is_ready()
        mem.tick(until_event=True)
        assert chunk.is_ready()

    t(hbm_mem)
    t(ddr4_mem)


def test_data_ready_timing_multi_chunk():
    hbm_mem, ddr4_mem, test_list = setup(128)

    def t(mem: MemSystem):
        gdl_window = int(mem.m_gdl_width / 4)
        mem.add_data_structure(test_list, 4)
        mem.mmap(0, 0, 0, 1, 0, data_index=0, length=len(test_list) * 4, offset=0)
        for i in range(8):
            chunk = mem[0, 0, 0, 1, i * gdl_window]
            assert not chunk.is_ready()
            mem.tick(until_event=True)
            print("should be empty", mem.nd_log[0][0][1][1])
            print(chunk)
            assert chunk.is_ready()

    t(ddr4_mem)
    t(hbm_mem)
