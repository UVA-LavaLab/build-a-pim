from lib.dramsim import callback_t
from lib.memsys import MemSystem
from lib.monad import DataStatus


def setup():
    # need to use HBM to test channels
    mem = MemSystem("./dramsim3/configs/HBM2_8Gb_x128.ini", ".")
    mem.set_pim_mode(True)
    return mem


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

    def add(channel: int, addr: int):
        return mem.add_transaction_to_bank(channel, 0, 0, 0, addr, False, True)

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
    print("Channels:", mem.c_num_channels)
    print("Ranks:", mem.c_num_ranks)
    print("Bankgroups per rank:", mem.c_num_bankgroups_per_rank)
    print("Banks per bg:", mem.c_num_banks_per_group)
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
    print(mem.c_num_banks)


if __name__ == "__main__":
    mem = MemSystem("./dramsim3/configs/DDR4_8Gb_x16_3200.ini", ".", nd_log=True)
    test_list = list(range(32))

    a = mem.bank_local_addr(0, 0, 0, 3, 4)
    print(mem.loc_from_addr(a))

    mem.add_data_structure(test_list, 4)
    # mem.mmap(0, 0, 0, 1, 0, data_index=0, length=len(test_list)*4, offset=4)
    mem.mmap(0, 0, 0, 1, 0, data_index=0, length=len(test_list)*4, offset=0)
    # print("start byte of data")
    # print(mem.start_byte_of_data(0, 0, 0, 0, 0))
    # print("now accessing structure")
    # d, b = mem.start_byte_of_data(0, 0, 0, 1, 4)
    # print(d, b)
    # print(mem.bank_access(0, 0, 0, 1, 0x4, 12))
    # print(mem.fetch_gdl_at(0, 0, 0, 1, 0x4))
    # print(mem.fetch_gdl_at(0, 0, 0, 1, 16))
    # print(mem.fetch_gdl_at(0, 0, 0, 1, 32))
    # print(mem.fetch_gdl_at(0, 0, 0, 1, 48))
    # print(mem.fetch_gdl_at(0, 0, 0, 1, 64))
    val = mem[0, 0, 0, 1, 0x10]
    val2 = None
    print(val)
    mem.toggle_pim_mode()
    # _ = mem.add_transaction_to_bank(0, 0, 0, 0, 0x0, False, True)
    # _ = mem.add_transaction_to_bank(0, 0, 0, 0, 0x1, False, True)
    for _ in range(2):
        mem.tick(until_event=True)
        print(mem.nd_log[0][0][0][1])
        if val.is_ready:
            print(val, mem.m_cycle)
            val2 = mem[0, 0, 0, 1, 0x0]
        if val2 is not None and val2.is_ready:
            print(val.is_ready)
            print(val, val2, mem.m_cycle)

    # print("Mem system created, toggling PIM mode.")
    # mem.toggle_pim_mode()
    # print("Toggled.")
    #
    # @callback_t
    # def log_cb(addr: int):
    #     global_reads_log.append((addr, mem.m_cycle + 1))
    #
    # @callback_t
    # def log_cb_w(addr: int):
    #     global_writes_log.append((addr, mem.m_cycle + 1))
    #
    # # mem.register_callbacks(log_cb, log_cb_w)
    #
    # print("created mem sys")
    # mem.print_stats()
    # print("print status called")
    # print(mem.m_cycle)
    # print("print cycle called")
    # print("should be true", mem.get_pim_mode())
    # mem.set_pim_mode(True)
    # print("should be true", mem.get_pim_mode())
    # mem.toggle_pim_mode()
    # mem.toggle_pim_mode()
    # print("should be true", mem.get_pim_mode())
    # mem.set_pim_mode(True)
    # print("set pim mode to True")
    # # _ = mem.add_transaction(0x0008, False)
    # # _ = mem.add_transaction(0x0001, False, True)
    # # _ = mem.add_transaction(0x0002, False, True)
    # # _ = mem.add_transaction(0x0003, False, True)
    # # _ = mem.add_transaction(0x0004, False, True)
    # _ = mem.add_transaction_to_bank(0, 0, 0, 0, 0x0002, is_write=False, is_pim=True)
    # mem.tick()
    # _ = mem.add_transaction_to_bank(0, 0, 0, 0, 0x0002, is_write=True, is_pim=True)
    # mem.tick()
    # _ = mem.add_transaction_to_bank(0, 0, 0, 0, 0x0002, is_write=False, is_pim=True)
    # mem.tick()
    # _ = mem.add_transaction_to_bank(0, 0, 0, 0, 0x01024, is_write=True, is_pim=True)
    # mem.tick()
    # _ = mem.add_transaction_to_bank(0, 0, 0, 0, 0x0002, is_write=False, is_pim=True)
    # # _ = mem.add_transaction_to_bank(0, 0, 0, 0, 0x0003, is_write=True, is_pim=True)
    # # _ = mem.add_transaction_to_bank(0, 0, 0, 1, 0x0001, is_write=True, is_pim=True)
    # # _ = mem.add_transaction_to_bank(0, 0, 0, 2, 0x0001, is_write=False, is_pim=True)
    # # _ = mem.add_transaction_to_bank(0, 0, 0, 3, 0x0001, is_write=True, is_pim=True)
    # print("transaction(s) added")
    # for i in range(100000):
    #     mem.tick()
    # print("Reads:", mem.m_reads)
    # print("Writes:", mem.m_writes)
    # print("Global Reads:", global_reads_log)
    # print("Global Writes:", global_writes_log)
    # print("ticking done")
    # print("ranks:", mem.c_num_ranks)
    # print("banks:", mem.c_num_banks_per_group)
    # print("channels:", mem.c_num_channels)
    # print("bankgroups:", mem.c_num_bankgroups_per_rank)
    # print("LOG READ:", mem.m_reads)
    # print("LOG WRITES:", mem.m_writes)
    # mem.destroy()
    # print("destroy called")
    # test_alternating_reads_bank_remote()
