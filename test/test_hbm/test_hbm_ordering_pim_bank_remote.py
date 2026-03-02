from lib.memsys import MemSystem
from lib.dramsim import callback_t, CallbackType, dramsim3


def setup():
    mem = MemSystem("./dramsim3/configs/HBM2_8Gb_x128.ini", ".")
    mem.set_pim_mode(True)
    return mem


def mask_addr(addr: int):
    return addr & 0b11111111


def test_r_to_w_ordering_pim_bank_remote():
    global_reads_log = []
    global_writes_log = []
    mem = setup()

    @callback_t
    def log_cb(addr: int):
        global_reads_log.append((addr, mem.m_cycle + 1))

    @callback_t
    def log_cb_w(addr: int):
        global_writes_log.append((addr, mem.m_cycle + 1))

    mem.register_callbacks(log_cb, log_cb_w)

    t1 = mem.add_transaction_to_bank(0, 0, 0, 1, 0x0, False, True)
    assert t1
    t1 = mem.add_transaction_to_bank(0, 0, 0, 0, 0x0, False, True)
    mem.tick()
    t2 = mem.add_transaction_to_bank(0, 0, 0, 1, 0x0, True, True)
    assert t2
    t2 = mem.add_transaction_to_bank(0, 0, 0, 0, 0x0, True, True)
    assert t1
    assert t2

    for i in range(1000):
        mem.tick()
        assert len(global_reads_log) >= len(global_writes_log)

    assert len(global_reads_log) > 1
    assert len(global_writes_log) > 1
    assert global_reads_log[0][1] < global_writes_log[0][1]
    assert global_reads_log[1][1] < global_writes_log[1][1]
    assert global_reads_log[0][1] == global_reads_log[1][1]
    assert global_writes_log[0][1] == global_writes_log[1][1]


def test_w_to_r_ordering_pim_bank_remote():
    global_reads_log = []
    global_writes_log = []
    mem = setup()

    @callback_t
    def log_cb(addr: int):
        global_reads_log.append((addr, mem.m_cycle + 1))

    @callback_t
    def log_cb_w(addr: int):
        global_writes_log.append((addr, mem.m_cycle + 1))

    mem.register_callbacks(log_cb, log_cb_w)

    t1 = mem.add_transaction_to_bank(0, 0, 0, 1, 0x0, True, True)
    assert t1
    t1 = mem.add_transaction_to_bank(0, 0, 0, 0, 0x0, True, True)
    mem.tick()
    t2 = mem.add_transaction_to_bank(0, 0, 0, 1, 0x0, False, True)
    assert t2
    t2 = mem.add_transaction_to_bank(0, 0, 0, 0, 0x0, False, True)
    assert t1
    assert t2

    for i in range(1000):
        mem.tick()
        assert len(global_reads_log) <= len(global_writes_log)

    assert len(global_reads_log) > 0
    assert len(global_writes_log) > 0
    assert global_reads_log[0][1] > global_writes_log[0][1]

    assert len(global_writes_log) > 1
    assert len(global_reads_log) > 1
    assert global_writes_log[0][1] < global_reads_log[0][1]
    assert global_writes_log[1][1] < global_reads_log[1][1]
    assert global_writes_log[0][1] == global_writes_log[1][1]
    assert global_reads_log[0][1] == global_reads_log[1][1]


def test_multiple_read_ordering_pim_bank_remote():
    global_reads_log = []
    global_writes_log = []
    mem = setup()

    @callback_t
    def log_cb(addr: int):
        global_reads_log.append((addr, mem.m_cycle + 1))

    @callback_t
    def log_cb_w(addr: int):
        global_writes_log.append((addr, mem.m_cycle + 1))

    def add(bank: int, addr: int):
        return mem.add_transaction_to_bank(0, 0, 0, bank, addr, False, True)

    mem.register_callbacks(log_cb, log_cb_w)

    t1 = add(0, 0x0)
    assert t1
    t1 = add(1, 0x0)
    assert t1
    mem.tick()
    t2 = add(0, 0x1)
    assert t2
    t2 = add(1, 0x1)
    assert t2
    mem.tick()
    t3 = add(0, 0x2)
    assert t3
    t3 = add(1, 0x2)
    assert t3
    mem.tick()
    t4 = add(0, 0x3)
    assert t4
    t4 = add(1, 0x3)
    assert t4

    for i in range(1000):
        mem.tick()

    print(global_reads_log)
    assert len(global_reads_log) == 8
    # test bank 1 first
    assert global_reads_log[0][0] == (1 << 14)
    assert global_reads_log[2][0] == (1 << 6) + (1 << 14)
    assert global_reads_log[4][0] == (2 << 6) + (1 << 14)
    assert global_reads_log[6][0] == (3 << 6) + (1 << 14)
    # now bank 0
    assert global_reads_log[1][0] == 0
    assert global_reads_log[3][0] == 1 << 6
    assert global_reads_log[5][0] == 2 << 6
    assert global_reads_log[7][0] == 3 << 6
    # now ensure that timings match
    assert global_reads_log[1][1] == global_reads_log[0][1]
    assert global_reads_log[3][1] == global_reads_log[2][1]
    assert global_reads_log[5][1] == global_reads_log[4][1]
    assert global_reads_log[7][1] == global_reads_log[6][1]


def test_multiple_write_ordering_pim_bank_remote():
    rl = []
    wl = []
    mem = setup()

    @callback_t
    def log_cb(addr: int):
        rl.append((addr, mem.m_cycle + 1))

    @callback_t
    def log_cb_w(addr: int):
        wl.append((addr, mem.m_cycle + 1))

    def add0(bank: int, addr: int):
        return mem.add_transaction_to_bank(0, 0, 0, bank, addr, True, True)

    mem.register_callbacks(log_cb, log_cb_w)

    t1 = add0(1, 0x0)
    assert t1
    t1 = add0(0, 0x0)
    assert t1
    mem.tick()
    t2 = add0(1, 0x1)
    assert t2
    t2 = add0(0, 0x1)
    assert t2
    mem.tick()
    t3 = add0(1, 0x2)
    assert t3
    t3 = add0(0, 0x2)
    assert t3
    mem.tick()
    t4 = add0(1, 0x3)
    assert t4
    t4 = add0(0, 0x3)
    assert t4

    for i in range(1000):
        mem.tick()

    print(wl)
    assert len(wl) == 8
    # test bank 1 first
    assert wl[0][0] == (1 << 14)
    assert wl[2][0] == (1 << 6) + (1 << 14)
    assert wl[4][0] == (2 << 6) + (1 << 14)
    assert wl[6][0] == (3 << 6) + (1 << 14)
    # now bank 0
    assert wl[1][0] == 0
    assert wl[3][0] == 1 << 6
    assert wl[5][0] == 2 << 6
    assert wl[7][0] == 3 << 6
    # now ensure that timings match
    assert wl[1][1] == wl[0][1]
    assert wl[3][1] == wl[2][1]
    assert wl[5][1] == wl[4][1]
    assert wl[7][1] == wl[6][1]


def test_alternating_reads_bank_remote():
    grl = []
    gwl = []
    mem = setup()

    @callback_t
    def log_cb(addr: int):
        grl.append((addr, mem.m_cycle + 1))

    @callback_t
    def log_cb_w(addr: int):
        gwl.append((addr, mem.m_cycle + 1))

    def add(bank: int, addr: int):
        return mem.add_transaction_to_bank(0, 0, 0, bank, addr, False, True)

    mem.register_callbacks(log_cb, log_cb_w)

    for i in range(128):
        v = add(0, i % 2)
        assert v
        v = add(1, i % 2)
        assert v
        mem.tick()

    i = 0
    while i < 1000000 and len(grl) < 128:
        mem.tick()

    print(grl)
    assert len(grl) == 128
    for i in range(int(len(grl) / 2)):
        print(
            "checking index",
            i,
            "local:",
            grl[2 * i],
            "remote:",
            grl[2 * i + 1],
            "expected:",
            (i % 2) << 6,
        )
        assert grl[2 * i][0] & 0b11111111 == (i % 2) << 6
        assert grl[2 * i + 1][0] & 0b11111111 == (i % 2) << 6
        assert grl[2 * i][1] == grl[2 * i + 1][1]


def test_alternating_writes():
    rl = []
    wl = []
    mem = setup()

    @callback_t
    def log_cb(addr: int):
        rl.append((addr, mem.m_cycle + 1))

    @callback_t
    def log_cb_w(addr: int):
        wl.append((addr, mem.m_cycle + 1))

    def addb(bank: int, addr: int):
        return mem.add_transaction_to_bank(0, 0, 0, bank, addr, True, True)

    mem.register_callbacks(log_cb, log_cb_w)

    for i in range(128):
        assert addb(0, i % 2)
        assert addb(1, i % 2)
        mem.tick()

    i = 0
    while i < 1000000 and len(wl) < 128:
        mem.tick()

    print(wl)
    assert len(wl) == 128
    for i in range(int(len(wl) / 2)):
        print(
            "checking index",
            i,
            "local:",
            wl[2 * i],
            "remote:",
            wl[2 * i + 1],
            "expected:",
            (i % 2) << 6,
        )
        assert mask_addr(wl[2 * i][0]) == (i % 2) << 6
        assert mask_addr(wl[2 * i + 1][0]) == (i % 2) << 6
        assert mask_addr(wl[2 * i][1]) == mask_addr(wl[2 * i + 1][1])


def test_alternating_read_then_writes_same_addr_bank_remote():
    rl = []
    wl = []
    mem = setup()

    @callback_t
    def log_cb(addr: int):
        rl.append((addr, mem.m_cycle + 1))

    @callback_t
    def log_cb_w(addr: int):
        wl.append((addr, mem.m_cycle + 1))

    def addb(bank: int, addr: int, write: bool):
        return mem.add_transaction_to_bank(0, 0, 0, bank, addr, not write, True)

    mem.register_callbacks(log_cb, log_cb_w)

    for i in range(128):
        assert addb(0, 0, i % 2 == 0)
        assert addb(1, 0, i % 2 == 0)
        mem.tick()

    i = 0
    while i < 1000000 and (len(wl) < 128 or len(rl) < 128):
        mem.tick()

    print("reads:", rl)
    print("writes:", wl)
    assert len(wl) == len(rl)
    for i in range(len(wl)):
        print(
            "checking index",
            i,
            "read:",
            rl[i][1],
            "write:",
            wl[i][1],
        )
        assert rl[i][1] < wl[i][1]
        if i < len(wl) / 2:
            assert rl[2 * i][1] == rl[2 * i + 1][1]
            assert wl[2 * i][1] == wl[2 * i + 1][1]
