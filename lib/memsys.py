import ctypes
from collections.abc import Callable
from tracemalloc import start
from typing import Any
from lib.address.address_mapper import AddressMapper
from lib.dramsim import callback_t, CallbackType, dramsim3
from lib.monad import DataStructureContainer, DataWrapper, DataSetter
from lib.types import Location
from lib.errors import MisalignedMemWriteError, PimAccessOutOfBoundsError
import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray


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
        self.address_mapper: AddressMapper = AddressMapper()
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

        self.m_memsys_ptr: ctypes.c_void_p = dramsim3.memsys_create(
            config.encode("ascii"),
            output.encode("ascii"),
            self.log_reads,
            self.log_writes,
        )
        self.m_destroyed: bool = False

        # start in PIM mode
        dramsim3.memsys_toggle_mode(self.m_memsys_ptr)

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
                channel, rank, bankgroup, bank, local_addr = self.loc_from_addr(addr)
                self.nd_log[channel][rank][bankgroup][bank].append((local_addr, False))
                self.event = True

            self.log_cb = log_cb
            dramsim3.memsys_register_callbacks(
                self.m_memsys_ptr, self.log_cb, self.log_cb
            )

    def get(
        self,
        addr: int | tuple[int, int, int, int, int] | Location,
        dtype: npt.DTypeLike = np.int32,
    ) -> DataWrapper:
        if isinstance(addr, int):
            channel, rank, bankgroup, bank, hex_addr = self.loc_from_addr(addr)
        else:
            channel, rank, bankgroup, bank, hex_addr = addr

        _ = self.add_transaction_to_bank(
            channel, rank, bankgroup, bank, hex_addr, is_write=False, is_pim=True
        )

        if self.nd_log:

            def update():
                if (
                    len(self.nd_log[channel][rank][bankgroup][bank]) > 0
                    and self.nd_log[channel][rank][bankgroup][bank][0][0] == hex_addr
                ):
                    _ = self.nd_log[channel][rank][bankgroup][bank].pop()
                    return True
                return False

        else:

            def update():
                return True

        item = DataWrapper(
            self.fetch_gdl_at(channel, rank, bankgroup, bank, hex_addr, dtype=dtype),
            update,
        )

        if len(item.data) == 0:
            raise Exception(f"item of length 0 {str(item)}, {item.data}")
        return item

    def set(
        self,
        addr: int | tuple[int, int, int, int, int] | Location,
        item: DataWrapper,
    ):
        if isinstance(addr, int):
            channel, rank, bankgroup, bank, hex_addr = self.loc_from_addr(addr)
        else:
            channel, rank, bankgroup, bank, hex_addr = addr

        _ = self.add_transaction_to_bank(
            channel, rank, bankgroup, bank, hex_addr, is_write=True, is_pim=True
        )

        if self.nd_log:

            def update():
                if (
                    len(self.nd_log[channel][rank][bankgroup][bank]) > 0
                    and self.nd_log[channel][rank][bankgroup][bank][0][0] == hex_addr
                ):
                    _ = self.nd_log[channel][rank][bankgroup][bank].pop()
                    self.bank_write(channel, rank, bankgroup, bank, hex_addr, item)
                    return True
                return False

        else:

            def update():
                return True

        return DataWrapper(np.array(item.data), update)

    def local_to_canonical_addr(
        self, location: tuple[int, int, int, int], addr: int
    ) -> int:
        return dramsim3.memsys_get_canonical_from_phys(
            self.m_memsys_ptr, location[0], location[1], location[2], location[3], addr
        )

    # def of_canoncial_addr(self, addr: int) -> tuple[int, int, int, int, int]:
    #     channel = ctypes.c_int64(0)
    #     rank = ctypes.c_int64(0)
    #     bankgroup = ctypes.c_int64(0)
    #     bank = ctypes.c_int64(0)
    #     local_addr = ctypes.c_int64(0)
    #     dramsim3.memsys_get_phys_from_canonical(
    #         self.m_memsys_ptr,
    #         ctypes.byref(channel),
    #         ctypes.byref(rank),
    #         ctypes.byref(bankgroup),
    #         ctypes.byref(bank),
    #         ctypes.byref(local_addr),
    #         addr,
    #     )
    #     return int(channel), int(rank), int(bankgroup), int(bank), int(local_addr)

    def get_gdl_bin(self, local_addr: int) -> int:
        return local_addr

    def get_config_param(self, id: str) -> int:
        c_id = ctypes.c_char_p(id.encode())
        return dramsim3.memsys_get_config_property(self.m_memsys_ptr, c_id)

    def get_active_row(self, channel: int, rank: int, bankgroup: int, bank: int) -> int:
        return dramsim3.memsys_get_active_row(
            self.m_memsys_ptr, channel, rank, bankgroup, bank
        )

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
        return dramsim3.memsys_get_cycle(self.m_memsys_ptr)

    @property
    def c_num_ranks(self) -> int:
        return dramsim3.memsys_get_ranks(self.m_memsys_ptr)

    @property
    def c_num_banks_per_group(self) -> int:
        return dramsim3.memsys_get_banks_per_bankgroup(self.m_memsys_ptr)

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
        return dramsim3.memsys_get_channels(self.m_memsys_ptr)

    @property
    def c_num_bankgroups_per_rank(self) -> int:
        return dramsim3.memsys_get_bankgroups_per_rank(self.m_memsys_ptr)

    @property
    def c_tck(self) -> np.float32:
        cf_tck = dramsim3.memsys_get_tck(self.m_memsys_ptr)
        return np.float32(cf_tck)

    def bank_local_addr(
        self, channel: int, rank: int, bankgroup: int, bank: int, hex_addr: int
    ) -> int:
        return dramsim3.memsys_get_address_from_physical_location(
            self.m_memsys_ptr, channel, rank, bankgroup, bank, hex_addr
        )

    def loc_from_addr(self, addr: int) -> tuple[int, int, int, int, int]:
        channel = ctypes.c_int64(0)
        rank = ctypes.c_int64(0)
        bankgroup = ctypes.c_int64(0)
        bank = ctypes.c_int64(0)
        local_addr = ctypes.c_int64(0)
        dramsim3.memsys_get_physical_location_from_address(
            self.m_memsys_ptr,
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

    def add_data_structure(
        self, data_structure: NDArray[np.generic] | list[Any]
    ) -> int:
        if isinstance(data_structure, list):
            data_structure = np.array(data_structure, dtype=np.int32)
        self.stored_data_structures.append(DataStructureContainer(data_structure))
        return len(self.stored_data_structures) - 1

    def get_num_data_structures(self) -> int:
        return len(self.stored_data_structures)

    def shift_offset(self, offset: int) -> int:
        return offset >> self.get_config_param("shift_bits")

    def fetch_gdl_at(
        self,
        channel: int,
        rank: int,
        bankgroup: int,
        bank: int,
        hex_addr: int,
        dtype: npt.DTypeLike = np.int32,
    ) -> NDArray[np.generic]:
        return self.bank_read(
            channel,
            rank,
            bankgroup,
            bank,
            hex_addr,
            int(self.get_config_param("gdl_width") / 8),
            dtype=dtype,
        )

    def bank_write(
        self,
        channel: int,
        rank: int,
        bankgroup: int,
        bank: int,
        hex_addr: int,
        data: DataWrapper,
    ):
        addr: int = self.local_to_canonical_addr(
            (channel, rank, bankgroup, bank), hex_addr
        )
        d, b = self.address_mapper[addr]

        if d == -1:
            raise PimAccessOutOfBoundsError(
                f"PIM access occurred out of bounds at canonical address {addr} (chan:{channel}, rank:{rank}, bg:{bankgroup}, bank:{bank})"
            )

        ds = self.stored_data_structures[d]
        gdl_width_bytes = int(self.m_gdl_width / 8)

        data_uint8 = np.frombuffer(data.data, dtype=np.uint8)
        # memcpy the bytes from our input array
        # into the stored datastructure
        np.frombuffer(
            ds.data_structure,
            dtype=np.uint8,
            offset=b * gdl_width_bytes,
            count=len(data_uint8),
        )[0 : len(data_uint8)] = data_uint8

    def bank_read(
        self,
        channel: int,
        rank: int,
        bankgroup: int,
        bank: int,
        hex_addr: int,
        length: int,
        dtype: npt.DTypeLike = np.int32,
    ) -> NDArray[np.generic]:
        addr: int = self.local_to_canonical_addr(
            (channel, rank, bankgroup, bank), hex_addr
        )
        d, b = self.address_mapper[addr]
        if d == -1:
            raise PimAccessOutOfBoundsError(
                f"PIM access occurred out of bounds at canonical address {addr} (chan:{channel}, rank:{rank}, bg:{bankgroup}, bank:{bank})"
            )

        if length == 0:
            raise Exception(
                f"Tried to read length 0 at addr {hex_addr}\nbank: {bank}\nbankgroup: {bankgroup}\nrank: {rank}\nchannel: {channel}\nstart byte: {b}\ndata index: {d}"
            )

        ds = self.stored_data_structures[d]
        gdl_width_bytes = int(self.m_gdl_width / 8)
        result = np.copy(
            np.frombuffer(
                np.frombuffer(ds.data_structure, dtype=np.uint8)[
                    b * gdl_width_bytes : b * gdl_width_bytes + length
                ],
                dtype=dtype,
            )
        )
        if len(result) == 0:
            raise Exception(
                f"Extracted result of length 0: {result} at addr 0d{hex_addr}\nbank: {bank}\nbankgroup: {bankgroup}\nrank: {rank}\nchannel: {channel}\nstart byte: {b}\ndata index: {d}\nstored datastructure:\n{ds}\nwith len: {len(np.frombuffer(ds.data_structure, dtype=np.uint8))}"
            )
        return result

    def start_byte_of_data(
        self, channel: int, rank: int, bankgroup: int, bank: int, hex_addr: int
    ) -> tuple[int, int]:
        start_idx = ctypes.c_size_t(0)
        data_idx = ctypes.c_int64(0)
        dramsim3.memsys_get_byte_range_from_bank(
            self.m_memsys_ptr,
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
        start: int = self.local_to_canonical_addr(
            (channel, rank, bankgroup, bank), hex_addr
        )
        self.address_mapper.add_mapping(start, start + length, data_index, offset)

    def munmap(
        self,
        channel: int,
        rank: int,
        bankgroup: int,
        bank: int,
        hex_addr: int,
        length: int,
    ):
        start: int = self.local_to_canonical_addr(
            (channel, rank, bankgroup, bank), hex_addr
        )
        self.address_mapper.remove_mapping(start, start + length)

    def print_config(self) -> None:
        print("Channels:", self.c_num_channels)
        print("Ranks:", self.c_num_ranks)
        print("Bankgroups per rank:", self.c_num_bankgroups_per_rank)
        print("Banks per bankgroup:", self.c_num_banks_per_group)

    def print_stats(self) -> None:
        dramsim3.memsys_print_stats(self.m_memsys_ptr)

    def register_callbacks(
        self, read_callback: CallbackType, write_callback: CallbackType
    ) -> None:
        dramsim3.memsys_register_callbacks(
            self.m_memsys_ptr, read_callback, write_callback
        )

    def tick(self, duration: int = 1, until_event: bool = False) -> None:
        if until_event:
            while not self.event:
                dramsim3.memsys_tick(self.m_memsys_ptr)
            self.event = False
            return
        for _ in range(duration):
            dramsim3.memsys_tick(self.m_memsys_ptr)

    def add_transaction(self, addr: int, is_write: bool, is_pim: bool = False) -> bool:
        return dramsim3.memsys_add_transaction(
            self.m_memsys_ptr, addr, is_write, is_pim
        )

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
            self.m_memsys_ptr, channel, rank, bankgroup, bank, addr, is_write, is_pim
        )

    def toggle_pim_mode(self) -> None:
        dramsim3.memsys_toggle_mode(self.m_memsys_ptr)

    def get_pim_mode(self) -> bool:
        return dramsim3.memsys_get_pim_mode(self.m_memsys_ptr)

    def set_pim_mode(self, mode: bool) -> None:
        dramsim3.memsys_set_pim_mode(self.m_memsys_ptr, mode)

    def destroy(self) -> None:
        self.m_destroyed = True
        dramsim3.memsys_destroy(self.m_memsys_ptr)

    # it's safer to call destroy yourself, but this
    # ideally cleans up your mess if you forget
    def __del__(self) -> None:
        if not self.m_destroyed:
            self.destroy()
