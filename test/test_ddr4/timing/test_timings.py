from lib.memsys import MemSystem
from lib.dramsim import callback_t, CallbackType, dramsim3


def setup():
    mem = MemSystem("./dramsim3/configs/DDR4_8Gb_x16_3200.ini", ".")
    mem.set_pim_mode(True)
    return mem


def test_sequential_read_intervals():
    rl = []
    wl = []
    mem = setup()

    @callback_t
    def log_cb(addr: int):
        rl.append((addr, mem.cycle + 1))

    @callback_t
    def log_cb_w(addr: int):
        wl.append((addr, mem.cycle + 1))

    def add(addr: int):
        return mem.add_transaction_to_bank(0, 0, 0, 0, addr, False, True)

    mem.register_callbacks(log_cb, log_cb_w)

    for i in range(16):
        assert add(i)

    for idx in range(1000):
        mem.tick()
        assert len(rl) >= 0
        assert len(wl) == 0

    assert len(rl) == 16
    print(rl)
    for fst, snd in zip(rl[:-1], rl[1:]):
        assert fst[1] == snd[1] - 8


def test_sequential_write_intervals():
    rl = []
    wl = []
    mem = setup()

    @callback_t
    def log_cb(addr: int):
        rl.append((addr, mem.cycle + 1))

    @callback_t
    def log_cb_w(addr: int):
        wl.append((addr, mem.cycle + 1))

    def add(addr: int):
        return mem.add_transaction_to_bank(0, 0, 0, 0, addr, True, True)

    mem.register_callbacks(log_cb, log_cb_w)

    for i in range(16):
        assert add(i)

    for idx in range(1000):
        mem.tick()
        assert len(wl) >= 0
        assert len(rl) == 0

    assert len(wl) == 16
    print(wl)
    for fst, snd in zip(wl[:-1], wl[1:]):
        assert fst[1] == snd[1] - 8


def test_sequential_alternating_read_write_intervals():
    rl = []
    wl = []
    mem = setup()

    @callback_t
    def log_cb(addr: int):
        rl.append((addr, mem.cycle + 1))

    @callback_t
    def log_cb_w(addr: int):
        wl.append((addr, mem.cycle + 1))

    def add(addr: int, write: bool):
        return mem.add_transaction_to_bank(0, 0, 0, 0, addr, write, True)

    mem.register_callbacks(log_cb, log_cb_w)

    for i in range(16):
        mem.tick()
        assert add(i, i % 2 == 1)

    for idx in range(2000):
        mem.tick()
        # assert len(rl) == len(wl) + 1 or len(rl) == len(wl)

    print("write list", wl)
    print("read list", rl)
    assert len(wl) == 8
    assert len(rl) == 8
    for fst, snd in zip(rl, wl):
        assert fst[1] == snd[1] - 8


def test_active_row_change_timing():
    rl = []
    wl = []
    mem = setup()

    @callback_t
    def log_cb(addr: int):
        rl.append((addr, mem.cycle + 1))

    @callback_t
    def log_cb_w(addr: int):
        wl.append((addr, mem.cycle + 1))

    def add(addr: int, write: bool):
        return mem.add_transaction_to_bank(0, 0, 0, 0, addr, write, True)

    mem.register_callbacks(log_cb, log_cb_w)

    for i in range(16):
        mem.tick()
        assert add((i % 2) * 8192, i % 2 == 1)

    for idx in range(2000):
        mem.tick()
        # assert len(rl) == len(wl) + 1 or len(rl) == len(wl)

    print("write list", wl)
    print("read list", rl)
    assert len(wl) == 8
    assert len(rl) == 8
    for fst, snd in zip(rl, wl):
        assert fst[1] == snd[1] - 74


def test_refresh_interrupt_timing():
    rl = []
    wl = []
    mem = setup()

    @callback_t
    def log_cb(addr: int):
        rl.append((addr, mem.cycle + 1))

    @callback_t
    def log_cb_w(addr: int):
        wl.append((addr, mem.cycle + 1))

    def add(bank: int, addr: int, write: bool):
        return mem.add_transaction_to_bank(0, 0, 0, bank, addr, write, True)

    mem.register_callbacks(log_cb, log_cb_w)

    assert add(0, 0x0, False)
    assert add(1, 0x0, True)

    mem.tick(6240)

    assert add(0, 0x0, False)
    assert add(1, 0x0, True)

    while len(wl) < 2 and len(rl) < 2:
        mem.tick()

    print("write list", wl)
    print("read list", rl)
    assert len(wl) == 2
    assert len(rl) == 2
    assert wl[1][1] > 6248
    assert rl[1][1] > 6248
    # for fst, snd in zip(rl, wl):
    #     assert fst[1] == snd[1] - 74
