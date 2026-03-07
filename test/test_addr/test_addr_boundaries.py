from lib.memsys import MemSystem
from lib.dramsim import callback_t, CallbackType, dramsim3


def count_1s(n: int) -> int:
    count = 0
    while n > 0:
        count += n % 2
        n = n >> 1
    return count


def setup():
    hbm_mem = MemSystem("./dramsim3/configs/HBM2_8Gb_x128.ini", ".", nd_log=True)
    hbm_mem.set_pim_mode(True)

    ddr4_mem = MemSystem("./dramsim3/configs/DDR4_8Gb_x16_3200.ini", ".", nd_log=True)
    ddr4_mem.set_pim_mode(True)

    return hbm_mem, ddr4_mem


def test_addr_positions():
    h, d = setup()

    def aux(mem: MemSystem):
        col = mem.get_config_param("co_pos")
        row = mem.get_config_param("ro_pos")
        bank = mem.get_config_param("ba_pos")
        bg = mem.get_config_param("bg_pos")
        rank = mem.get_config_param("ra_pos")
        channel = mem.get_config_param("ch_pos")
        shifts = [col, row, bank, bg, rank, channel]

        for x, i in enumerate(shifts):
            for y, j in enumerate(shifts):
                if i != j:
                    assert x != y

    aux(h)
    aux(d)


def test_row_bank_overlap():
    h, d = setup()

    def aux(mem: MemSystem):
        ro = mem.get_config_param("ro_mask")
        ba = mem.get_config_param("ba_mask")
        ro_pos = mem.get_config_param("ro_pos")
        ba_pos = mem.get_config_param("ba_pos")

        # test for inequality
        assert (ba << ba_pos) < (ro << ro_pos)
        # test for overlap
        assert ((ba << ba_pos) & (ro << ro_pos)) == 0

    aux(h)
    aux(d)


def test_col_row_overlap():
    h, d = setup()

    def aux(mem: MemSystem):
        ro = mem.get_config_param("ro_mask")
        co = mem.get_config_param("co_mask")
        ro_pos = mem.get_config_param("ro_pos")
        co_pos = mem.get_config_param("co_pos")

        # test for inequality
        assert (co << co_pos) < (ro << ro_pos)
        # test for overlap
        assert ((co << co_pos) & (ro << ro_pos)) == 0

    aux(h)
    aux(d)


def test_col_bank_overlap():
    h, d = setup()

    def aux(mem: MemSystem):
        ba = mem.get_config_param("ba_mask")
        co = mem.get_config_param("co_mask")
        ba_pos = mem.get_config_param("ba_pos")
        co_pos = mem.get_config_param("co_pos")

        # test for inequality
        assert (co << co_pos) < (ba << ba_pos)
        # test for overlap
        assert ((co << co_pos) & (ba << ba_pos)) == 0

    aux(h)
    aux(d)


def test_addr_bijection_col_bank():
    h, d = setup()

    def aux(mem: MemSystem):
        co = mem.get_config_param("co_mask")

        global_addr = mem.bank_local_addr(0, 0, 0, 1, co)
        _, _, _, b, local_addr = mem.loc_from_addr(global_addr)
        assert b == 1
        assert local_addr == co

    aux(h)
    aux(d)


def test_addr_bijection_row_bank():
    h, d = setup()

    def aux(mem: MemSystem):
        co = mem.get_config_param("co_mask")
        shift = count_1s(co)
        row_max = co << shift

        global_addr = mem.bank_local_addr(0, 0, 0, 1, row_max)
        _, _, _, b, local_addr = mem.loc_from_addr(global_addr)
        assert b == 1
        assert local_addr == row_max

    aux(h)
    aux(d)
