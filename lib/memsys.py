import ctypes
from collections.abc import Callable
from tracemalloc import start
from typing import Any
from lib.dramsim import callback_t, CallbackType, dramsim3
from lib.monad import DataStructureContainer, DataWrapper
import numpy as np


@callback_t
def do_nothing_cb(_: int):
    pass


@callback_t
def print_cb(addr: int):
    print(addr)


class MemSystem:
    def __init__(
        self,
        config: str,
        output: str,
        read_cb: CallbackType | None = None,
        write_cb: CallbackType | None = None,
        nd_log: bool = False,
    ) -> None:
        self.stored_data_structures: list[DataStructureContainer] = []
        self.m_reads: list[tuple[int, int]] = []
        self.m_writes: list[tuple[int, int]] = []
        self.event = False
        # has to be defined this way, explained later
        if read_cb is None:

            @callback_t
            def read_cb(addr: int):
                self.m_reads.append((self.shift_offset(addr), self.m_cycle + 1))
                self.event = True

        if write_cb is None:

            @callback_t
            def write_cb(addr: int):
                self.m_writes.append((self.shift_offset(addr), self.m_cycle + 1))
                self.event = True

        # this HAS to be an instance variable or
        # it is garbage collected after
        # being passed to the C++ wrapper
        self.log_reads: CallbackType = read_cb
        self.log_writes: CallbackType = write_cb

        self.m_memsys: ctypes.c_void_p = dramsim3.memsys_create(
            config.encode("ascii"),
            output.encode("ascii"),
            self.log_reads,
            self.log_writes,
        )
        self.m_destroyed: bool = False

        if nd_log:
            self.nd_log: list[list[list[list[list[tuple[int, bool]]]]]] = [
                [
                    [
                        [[] for _ in range(self.c_num_banks)]
                        for _ in range(self.c_num_bankgroups_per_rank)
                    ]
                    for _ in range(self.c_num_ranks)
                ]
                for _ in range(self.c_num_channels)
            ]

            @callback_t
            def log_cb(addr: int):
                print("translating", addr)
                channel, rank, bankgroup, bank, local_addr = self.loc_from_addr(addr)
                self.nd_log[channel][rank][bankgroup][bank].append((local_addr, False))
                print("Logging local address", local_addr)
                self.event = True

            self.log_cb = log_cb
            dramsim3.memsys_register_callbacks(self.m_memsys, self.log_cb, self.log_cb)

    def __getitem__(self, addr: int | tuple[int, int, int, int, int]) -> Any:
        if isinstance(addr, int):
            channel, rank, bankgroup, bank, hex_addr = self.loc_from_addr(addr)
        else:
            channel, rank, bankgroup, bank, hex_addr = addr

        _ = self.add_transaction_to_bank(
            channel, rank, bankgroup, bank, hex_addr, is_write=False, is_pim=True
        )

        if self.nd_log:

            def update():
                print("update time", self.nd_log[channel][rank][bankgroup][bank])
                if len(
                    self.nd_log[channel][rank][bankgroup][bank]
                ) > 0 and self.get_gdl_bin(
                    self.nd_log[channel][rank][bankgroup][bank][0][0]
                ) == self.get_gdl_bin(
                    hex_addr
                ):
                    print("updated, popping:", self.nd_log[channel][rank][bankgroup][bank])
                    _ = self.nd_log[channel][rank][bankgroup][bank].pop()
                    return True
                return False

        else:

            def update():
                return True

        return DataWrapper(
            self.fetch_gdl_at(channel, rank, bankgroup, bank, hex_addr), update
        )

    def get_gdl_bin(self, local_addr: int) -> int:
        return int(local_addr / (self.m_gdl_width / 8))

    def get_config_param(self, id: str) -> int:
        c_id = ctypes.c_char_p(id.encode())
        return dramsim3.memsys_get_config_property(self.m_memsys, c_id)

    @callback_t
    def record_reads(self, addr: ctypes.c_uint64):
        self.m_reads.append((int(addr), self.m_cycle + 1))

    @callback_t
    def record_writes(self, addr: ctypes.c_uint64):
        self.m_writes.append((int(addr), self.m_cycle + 1))

    @property
    def m_gdl_width(self) -> int:
        return self.get_config_param("gdl_width")

    @property
    def m_cycle(self) -> int:
        return dramsim3.memsys_get_cycle(self.m_memsys)

    @property
    def c_num_ranks(self) -> int:
        return dramsim3.memsys_get_ranks(self.m_memsys)

    @property
    def c_num_banks_per_group(self) -> int:
        return dramsim3.memsys_get_banks_per_bankgroup(self.m_memsys)

    @property
    def c_num_banks(self) -> int:
        return (
            self.c_num_bankgroups_per_rank
            * self.c_num_banks_per_group
            * self.c_num_ranks
            * self.c_num_channels
        )

    @property
    def c_num_channels(self) -> int:
        return dramsim3.memsys_get_channels(self.m_memsys)

    @property
    def c_num_bankgroups_per_rank(self) -> int:
        return dramsim3.memsys_get_bankgroups_per_rank(self.m_memsys)

    def bank_local_addr(
        self, channel: int, rank: int, bankgroup: int, bank: int, hex_addr: int
    ) -> int:
        return dramsim3.memsys_get_address_from_physical_location(
            self.m_memsys, channel, rank, bankgroup, bank, hex_addr
        )

    def loc_from_addr(self, addr: int) -> tuple[int, int, int, int, int]:
        channel = ctypes.c_int64(0)
        rank = ctypes.c_int64(0)
        bankgroup = ctypes.c_int64(0)
        bank = ctypes.c_int64(0)
        local_addr = ctypes.c_int64(0)
        dramsim3.memsys_get_physical_location_from_address(
            self.m_memsys,
            ctypes.byref(channel),
            ctypes.byref(rank),
            ctypes.byref(bankgroup),
            ctypes.byref(bank),
            ctypes.byref(local_addr),
            addr,
        )
        return (
            int(channel.value),
            int(rank.value),
            int(bankgroup.value),
            int(bank.value),
            int(local_addr.value),
        )

    def add_data_structure(self, data_structure: Any, element_size_bytes: int):
        self.stored_data_structures.append(
            DataStructureContainer(data_structure, element_size_bytes)
        )

    def shift_offset(self, offset: int) -> int:
        return offset >> self.get_config_param("shift_bits")

    def fetch_gdl_at(
        self,
        channel: int,
        rank: int,
        bankgroup: int,
        bank: int,
        hex_addr: int,
    ):
        return self.bank_access(
            channel,
            rank,
            bankgroup,
            bank,
            hex_addr - (hex_addr % int(self.m_gdl_width / 8)),
            int(self.get_config_param("gdl_width") / 8),
        )

    def bank_access(
        self,
        channel: int,
        rank: int,
        bankgroup: int,
        bank: int,
        hex_addr: int,
        length: int,
    ) -> Any:
        d, b = self.start_byte_of_data(channel, rank, bankgroup, bank, hex_addr)

        def get_data(key: tuple[int, int, int]) -> Any:
            ds = self.stored_data_structures[key[0]]
            size = ds.element_size_bytes
            if key[1] % size != 0 or key[2] % size != 0:
                raise Exception(
                    f"Misaligned memory access in data structure with ID {key[0]} and read bounds {key[1]} -> {key[2]}"
                )
            return ds.data_structure[int(key[1] / size) : int(key[2] / size)]

        return get_data((d, b, b + length))

    def start_byte_of_data(
        self, channel: int, rank: int, bankgroup: int, bank: int, hex_addr: int
    ) -> tuple[int, int]:
        start_idx = ctypes.c_size_t(0)
        data_idx = ctypes.c_int64(0)
        dramsim3.memsys_get_byte_range_from_bank(
            self.m_memsys,
            channel,
            rank,
            bankgroup,
            bank,
            hex_addr,
            ctypes.byref(data_idx),
            ctypes.byref(start_idx),
        )
        return (data_idx.value, start_idx.value)

    def mmap(
        self,
        channel: int,
        rank: int,
        bankgroup: int,
        bank: int,
        hex_addr: int,
        data_index: int,
        length: int,
        offset: int,
    ):
        c_data = ctypes.c_int64(data_index)
        c_off = ctypes.c_size_t(offset)
        c_len = ctypes.c_size_t(length)
        dramsim3.memsys_mmap(
            self.m_memsys,
            channel,
            rank,
            bankgroup,
            bank,
            hex_addr,
            c_data,
            c_len,
            c_off,
        )

    def munmap(
        self,
        channel: int,
        rank: int,
        bankgroup: int,
        bank: int,
        hex_addr: int,
        length: int,
    ):
        c_hex_addr = ctypes.c_size_t(hex_addr)
        c_len = ctypes.c_size_t(length)
        dramsim3.memsys_munmap(
            self.m_memsys, channel, rank, bankgroup, bank, c_hex_addr, c_len
        )

    # void MMap(int64_t data_index, size_t start_addr, size_t end_addr, size_t offset);

    def print_config(self) -> None:
        print("Channels:", self.c_num_channels)
        print("Ranks:", self.c_num_ranks)
        print("Bankgroups per rank:", self.c_num_bankgroups_per_rank)
        print("Banks per bankgroup:", self.c_num_banks_per_group)

    def print_stats(self) -> None:
        dramsim3.memsys_print_stats(self.m_memsys)

    def register_callbacks(
        self, read_callback: CallbackType, write_callback: CallbackType
    ) -> None:
        dramsim3.memsys_register_callbacks(self.m_memsys, read_callback, write_callback)

    def tick(self, duration: int = 1, until_event: bool = False) -> None:
        if until_event:
            while not self.event:
                dramsim3.memsys_tick(self.m_memsys)
            self.event = False
            return
        for _ in range(duration):
            dramsim3.memsys_tick(self.m_memsys)

    def add_transaction(self, addr: int, is_write: bool, is_pim: bool = False) -> bool:
        return dramsim3.memsys_add_transaction(self.m_memsys, addr, is_write, is_pim)

    def add_transaction_to_bank(
        self,
        channel: int,
        rank: int,
        bankgroup: int,
        bank: int,
        addr: int,
        is_write: bool,
        is_pim: bool,
    ) -> bool:
        return dramsim3.memsys_add_transaction_to_bank(
            self.m_memsys, channel, rank, bankgroup, bank, addr, is_write, is_pim
        )

    def toggle_pim_mode(self) -> None:
        dramsim3.memsys_toggle_mode(self.m_memsys)

    def get_pim_mode(self) -> bool:
        return dramsim3.memsys_get_pim_mode(self.m_memsys)

    def set_pim_mode(self, mode: bool) -> None:
        dramsim3.memsys_set_pim_mode(self.m_memsys, mode)

    def destroy(self) -> None:
        self.m_destroyed = True
        dramsim3.memsys_destroy(self.m_memsys)

    # it's safer to call destroy yourself, but this
    # ideally cleans up your mess if you forget
    def __del__(self) -> None:
        if not self.m_destroyed:
            self.destroy()
