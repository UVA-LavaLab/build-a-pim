from lib.memsys import MemSystem
from lib.dramsim import callback_t, CallbackType, dramsim3


def setup():
    mem = MemSystem("./dramsim3/configs/HBM2_8Gb_x128.ini", ".")
    mem.set_pim_mode(True)
    return mem


def test_r_to_w_ordering_pim():
    global_reads_log = []
    global_writes_log = []
    mem = setup()

    @callback_t
    def log_cb(addr: int):
        global_reads_log.append((addr, mem.cycle + 1))

    @callback_t
    def log_cb_w(addr: int):
        global_writes_log.append((addr, mem.cycle + 1))

    mem.register_callbacks(log_cb, log_cb_w)

    t1 = mem.add_transaction_to_bank(0, 0, 0, 0, 0x0, False, True)
    mem.tick()
    t2 = mem.add_transaction_to_bank(0, 0, 0, 0, 0x0, True, True)
    assert t1
    assert t2

    for i in range(1000):
        mem.tick()
        assert len(global_reads_log) >= len(global_writes_log)

    assert len(global_reads_log) > 0
    assert len(global_writes_log) > 0
    assert global_reads_log[0][1] < global_writes_log[0][1]


def test_w_to_r_ordering_pim():
    global_reads_log = []
    global_writes_log = []
    mem = setup()

    @callback_t
    def log_cb(addr: int):
        global_reads_log.append((addr, mem.cycle + 1))

    @callback_t
    def log_cb_w(addr: int):
        global_writes_log.append((addr, mem.cycle + 1))

    mem.register_callbacks(log_cb, log_cb_w)

    t1 = mem.add_transaction_to_bank(0, 0, 0, 0, 0x0, True, True)
    mem.tick()
    t2 = mem.add_transaction_to_bank(0, 0, 0, 0, 0x0, False, True)
    assert t1
    assert t2

    for i in range(1000):
        mem.tick()
        assert len(global_reads_log) <= len(global_writes_log)

    assert len(global_reads_log) > 0
    assert len(global_writes_log) > 0
    assert global_reads_log[0][1] > global_writes_log[0][1]


def test_multiple_read_ordering_pim():
    global_reads_log = []
    global_writes_log = []
    mem = setup()

    @callback_t
    def log_cb(addr: int):
        global_reads_log.append((addr, mem.cycle + 1))

    @callback_t
    def log_cb_w(addr: int):
        global_writes_log.append((addr, mem.cycle + 1))

    def add0(addr: int):
        return mem.add_transaction_to_bank(0, 0, 0, 0, addr, False, True)

    mem.register_callbacks(log_cb, log_cb_w)

    t1 = add0(0x0)
    mem.tick()
    t2 = add0(0x1)
    mem.tick()
    t3 = add0(0x2)
    mem.tick()
    t4 = add0(0x3)
    assert t1
    assert t2
    assert t3
    assert t4

    for i in range(1000):
        mem.tick()

    print(global_reads_log)
    assert len(global_reads_log) == 4
    assert global_reads_log[0][0] == 0
    assert global_reads_log[1][0] == 1 << 6
    assert global_reads_log[2][0] == 2 << 6
    assert global_reads_log[3][0] == 3 << 6


def test_multiple_write_ordering_pim():
    global_reads_log = []
    global_writes_log = []
    mem = setup()

    @callback_t
    def log_cb(addr: int):
        global_reads_log.append((addr, mem.cycle + 1))

    @callback_t
    def log_cb_w(addr: int):
        global_writes_log.append((addr, mem.cycle + 1))

    def add0(addr: int):
        return mem.add_transaction_to_bank(0, 0, 0, 0, addr, True, True)

    mem.register_callbacks(log_cb, log_cb_w)

    t1 = add0(0x0)
    mem.tick()
    t2 = add0(0x1)
    mem.tick()
    t3 = add0(0x2)
    mem.tick()
    t4 = add0(0x3)
    assert t1
    assert t2
    assert t3
    assert t4

    for i in range(1000):
        mem.tick()

    print(global_writes_log)
    assert len(global_writes_log) == 4
    assert global_writes_log[0][0] == 0
    assert global_writes_log[1][0] == 1 << 6
    assert global_writes_log[2][0] == 2 << 6
    assert global_writes_log[3][0] == 3 << 6


def test_alternating_reads():
    global_reads_log = []
    global_writes_log = []
    mem = setup()

    @callback_t
    def log_cb(addr: int):
        global_reads_log.append((addr, mem.cycle + 1))

    @callback_t
    def log_cb_w(addr: int):
        global_writes_log.append((addr, mem.cycle + 1))

    def add0(addr: int):
        return mem.add_transaction_to_bank(0, 0, 0, 0, addr, False, True)

    mem.register_callbacks(log_cb, log_cb_w)

    for i in range(128):
        v = add0(i % 2)
        mem.tick()
        assert v

    i = 0
    while i < 1000000 and len(global_reads_log) < 128:
        mem.tick()

    print(global_reads_log)
    assert len(global_reads_log) == 128
    for i in range(len(global_reads_log)):
        print(
            "checking index",
            i,
            "value:",
            global_reads_log[i][0],
            "expected:",
            (i % 2) << 6,
        )
        assert global_reads_log[i][0] == (i % 2) << 6


def test_alternating_writes():
    global_reads_log = []
    global_writes_log = []
    mem = setup()

    @callback_t
    def log_cb(addr: int):
        global_reads_log.append((addr, mem.cycle + 1))

    @callback_t
    def log_cb_w(addr: int):
        global_writes_log.append((addr, mem.cycle + 1))

    def add0(addr: int):
        return mem.add_transaction_to_bank(0, 0, 0, 0, addr, True, True)

    mem.register_callbacks(log_cb, log_cb_w)

    for i in range(128):
        v = add0(i % 2)
        mem.tick()
        assert v

    i = 0
    while i < 1000000 and len(global_writes_log) < 128:
        mem.tick()

    print(global_writes_log)
    assert len(global_writes_log) == 128
    for i in range(len(global_writes_log)):
        print(
            "checking index",
            i,
            "value:",
            global_writes_log[i][0],
            "expected:",
            (i % 2) << 6,
        )
        assert global_writes_log[i][0] == (i % 2) << 6


def test_alternating_read_then_writes_same_addr():
    global_reads_log = []
    global_writes_log = []
    mem = setup()

    @callback_t
    def log_cb(addr: int):
        global_reads_log.append((addr, mem.cycle + 1))

    @callback_t
    def log_cb_w(addr: int):
        global_writes_log.append((addr, mem.cycle + 1))

    def add0(addr: int, write: bool):
        return mem.add_transaction_to_bank(0, 0, 0, 0, addr, not write, True)

    mem.register_callbacks(log_cb, log_cb_w)

    for i in range(128):
        v = add0(0, i % 2 == 0)
        mem.tick()
        assert v

    i = 0
    while i < 1000000 and (len(global_writes_log) < 64 or len(global_reads_log) < 64):
        mem.tick()

    print("reads", global_reads_log)
    print("writes", global_writes_log)
    assert len(global_writes_log) == len(global_reads_log)
    for i in range(len(global_writes_log)):
        print(
            "checking index",
            i,
            "read:",
            global_reads_log[i][1],
            "write:",
            global_writes_log[i][1],
        )
        assert global_reads_log[i][1] < global_writes_log[i][1]
