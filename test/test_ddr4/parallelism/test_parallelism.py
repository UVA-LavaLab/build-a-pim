from lib.memsys import MemSystem
from lib.dramsim import callback_t, CallbackType, dramsim3


def setup():
    mem = MemSystem("./dramsim3/configs/DDR4_8Gb_x16_3200.ini", ".")
    mem.set_pim_mode(True)
    return mem


def test_bank_parallel_read():
    rl = []
    wl = []
    mem = setup()

    @callback_t
    def log_cb(addr: int):
        rl.append((addr, mem.m_cycle + 1))

    @callback_t
    def log_cb_w(addr: int):
        wl.append((addr, mem.m_cycle + 1))

    def add(bank: int, addr: int):
        return mem.add_transaction_to_bank(0, 0, 0, bank, addr, False, True)

    mem.register_callbacks(log_cb, log_cb_w)

    for bank in range(4):
        assert add(bank, 0x0)

    for i in range(1000):
        mem.tick()
        assert len(rl) >= 0
        assert len(wl) == 0

    all_matching = True
    default = rl[0][1]
    for r in rl:
        all_matching = all_matching and default == r[1]
    assert all_matching


def test_multilevel_parallel_read():
    rl = []
    wl = []
    mem = setup()

    @callback_t
    def log_cb(addr: int):
        rl.append((addr, mem.m_cycle + 1))

    @callback_t
    def log_cb_w(addr: int):
        wl.append((addr, mem.m_cycle + 1))

    def add(channel: int, rank: int, bankgroup: int, bank: int, addr: int):
        return mem.add_transaction_to_bank(
            channel, rank, bankgroup, bank, addr, False, True
        )

    mem.register_callbacks(log_cb, log_cb_w)

    for bank in range(mem.c_num_banks_per_group):
        for bankgroup in range(mem.c_num_bankgroups_per_rank):
            for rank in range(mem.c_num_ranks):
                for channel in range(mem.c_num_channels):
                    # access different local addresses in each bank
                    idx: int = (
                        mem.c_num_ranks
                        * mem.c_num_bankgroups_per_rank
                        * mem.c_num_banks_per_group
                        * channel
                        + mem.c_num_bankgroups_per_rank
                        * mem.c_num_banks_per_group
                        * rank
                        + mem.c_num_banks_per_group * bankgroup
                        + bank
                    )
                    assert add(channel, rank, bankgroup, bank, idx)

    for idx in range(1000):
        mem.tick()
        assert len(rl) >= 0
        assert len(wl) == 0

    assert (
        len(rl)
        == mem.c_num_bankgroups_per_rank
        * mem.c_num_banks_per_group
        * mem.c_num_ranks
        * mem.c_num_channels
    )
    print(rl)
    all_matching = True
    default = rl[0][1]
    for pair in rl:
        all_matching = all_matching and default == pair[1]
    assert all_matching


def test_multilevel_parallel_write():
    rl = []
    wl = []
    mem = setup()

    @callback_t
    def log_cb(addr: int):
        rl.append((addr, mem.m_cycle + 1))

    @callback_t
    def log_cb_w(addr: int):
        wl.append((addr, mem.m_cycle + 1))

    def add(channel: int, rank: int, bankgroup: int, bank: int, addr: int):
        return mem.add_transaction_to_bank(
            channel, rank, bankgroup, bank, addr, True, True
        )

    mem.register_callbacks(log_cb, log_cb_w)

    for bank in range(mem.c_num_banks_per_group):
        for bankgroup in range(mem.c_num_bankgroups_per_rank):
            for rank in range(mem.c_num_ranks):
                for channel in range(mem.c_num_channels):
                    # access different local addresses in each bank
                    idx: int = (
                        mem.c_num_ranks
                        * mem.c_num_bankgroups_per_rank
                        * mem.c_num_banks_per_group
                        * channel
                        + mem.c_num_bankgroups_per_rank
                        * mem.c_num_banks_per_group
                        * rank
                        + mem.c_num_banks_per_group * bankgroup
                        + bank
                    )
                    assert add(channel, rank, bankgroup, bank, idx)

    for idx in range(1000):
        mem.tick()
        assert len(wl) >= 0
        assert len(rl) == 0

    assert (
        len(wl)
        == mem.c_num_bankgroups_per_rank
        * mem.c_num_banks_per_group
        * mem.c_num_ranks
        * mem.c_num_channels
    )
    print(wl)
    all_matching = True
    default = wl[0][1]
    for pair in wl:
        all_matching = all_matching and default == pair[1]
    assert all_matching


def test_multilevel_parallel_read_write():
    rl = []
    wl = []
    mem = setup()

    @callback_t
    def log_cb(addr: int):
        rl.append((addr, mem.m_cycle + 1))

    @callback_t
    def log_cb_w(addr: int):
        wl.append((addr, mem.m_cycle + 1))

    def add(channel: int, rank: int, bankgroup: int, bank: int, addr: int):
        return mem.add_transaction_to_bank(
            channel,
            rank,
            bankgroup,
            bank,
            addr,
            (channel + rank + bankgroup + bank) % 2 == 0,
            True,
        )

    mem.register_callbacks(log_cb, log_cb_w)

    for bank in range(mem.c_num_banks_per_group):
        for bankgroup in range(mem.c_num_bankgroups_per_rank):
            for rank in range(mem.c_num_ranks):
                for channel in range(mem.c_num_channels):
                    # access different local addresses in each bank
                    idx: int = (
                        mem.c_num_ranks
                        * mem.c_num_bankgroups_per_rank
                        * mem.c_num_banks_per_group
                        * channel
                        + mem.c_num_bankgroups_per_rank
                        * mem.c_num_banks_per_group
                        * rank
                        + mem.c_num_banks_per_group * bankgroup
                        + bank
                    )
                    assert add(channel, rank, bankgroup, bank, idx)

    for idx in range(1000):
        mem.tick()

    assert (
        len(wl) + len(rl)
        == mem.c_num_bankgroups_per_rank
        * mem.c_num_banks_per_group
        * mem.c_num_ranks
        * mem.c_num_channels
    )
    print(wl)
    all_matching = True
    joined = wl + rl
    default = joined[0][1]
    for pair in joined:
        all_matching = all_matching and default == pair[1]
    assert all_matching
