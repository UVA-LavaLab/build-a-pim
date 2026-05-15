import ctypes
from typing import Any
from lib.address.address_mapper import AddressMapper
from lib.dramsim import callback_t, CallbackType, dramsim3
from lib.containers import DataStructureContainer, Box 
from lib.types import Location
from lib.errors import PimAccessOutOfBoundsError
import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray


class MemSystem:
    """
    The point of integration between Build-A-PIM and our DRAMsim3 extension. It
    has two modes of recording memory events: a one-dimensional list and a
    4-dimensional list (channel x rank x bankgroup x bank). The 4-dimensional
    list is enabled by default (ng_log=True).

    This functionality can be entirely replaced by passing a read or write callback.

    The config argument should be a path relative to the runtime directory
    which points to a valid DRAMsim3 configuration. The device can be accessed
    using get() and set() and will automatically garbage collect its C++
    backend on Python garbage collection.

    The output argument is the path where DRAMsim3 logs should be output if you
    compiled with thermal or command tracing enabled.

    All addresses are in GDL-width chunks, not bytes. This helps avoid a
    significant amount of redundant math, since most PIM devices are SIMD. In
    the event that you need more granular addressing, you can add an
    intermediate address management solution which right-shifts by log2(gdl
    width in bytes) before accessing the device to achieve byte-level
    addressing.
    """
    def __init__(
        self,
        config: str,
        output: str = ".",
        read_cb: CallbackType | None = None,
        write_cb: CallbackType | None = None,
        nd_log: bool = True,
    ) -> None:
        self.stored_data_structures: list[DataStructureContainer] = []
        self.address_mapper: AddressMapper = AddressMapper()
        self.reads: list[tuple[int, int]] = []
        self.writes: list[tuple[int, int]] = []
        self.event = False
        # has to be defined this way, explained later
        if read_cb is None:

            @callback_t
            def read_cb(addr: int):
                self.reads.append((self.shift_offset(addr), self.cycle + 1))
                self.event = True

        if write_cb is None:

            @callback_t
            def write_cb(addr: int):
                self.writes.append((self.shift_offset(addr), self.cycle + 1))
                self.event = True

        # this HAS to be an instance variable or
        # it is garbage collected after
        # being passed to the C++ wrapper
        self.log_reads: CallbackType = read_cb
        self.log_writes: CallbackType = write_cb

        self.memsys_ptr: ctypes.c_void_p = dramsim3.memsys_create(
            config.encode("ascii"),
            output.encode("ascii"),
            self.log_reads,
            self.log_writes,
        )
        self.destroyed: bool = False

        # start in PIM mode
        dramsim3.memsys_toggle_mode(self.memsys_ptr)
        self._params: dict[str, int] = {}

        if nd_log:
            self.nd_log: list[list[list[list[list[tuple[int, bool]]]]]] = [
                [
                    [
                        [[] for _ in range(self.num_banks)]
                        for _ in range(self.num_bankgroups_per_rank)
                    ]
                    for _ in range(self.num_ranks)
                ]
                for _ in range(self.num_channels)
            ]

            @callback_t
            def log_cb(addr: int):
                channel, rank, bankgroup, bank, local_addr = self.loc_from_addr(addr)
                self.nd_log[channel][rank][bankgroup][bank].append((local_addr, False))
                self.event = True

            self.log_cb = log_cb
            dramsim3.memsys_register_callbacks(
                self.memsys_ptr, self.log_cb, self.log_cb
            )

    def get(
        self,
        addr: int | tuple[int, int, int, int, int] | Location,
        dtype: npt.DTypeLike = np.int32,
    ) -> Box:
        """
        The main way to access data within the MemSystem class from
        a PIM core. addr is a tuple of the form (channel, rank,
        bankgroup, bank, local address).

        This returns a Box with appropriate timing metadata. To
        update the returned Box, you *must* call Box.is_ready().
        """
        if isinstance(addr, int):
            channel, rank, bankgroup, bank, hex_addr = self.loc_from_addr(addr)
        else:
            channel, rank, bankgroup, bank, hex_addr = addr

        accepted = self.add_transaction_to_bank(
            channel, rank, bankgroup, bank, hex_addr, is_write=False, is_pim=True
        )
        assert accepted

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

        item = Box(
            self.fetch_gdl_at(channel, rank, bankgroup, bank, hex_addr, dtype=dtype),
            update,
        )

        # TODO: determine if this check should be omitted or just relaxed
        # if len(item.data) == 0:
        #     raise Exception(f"item of length 0 {str(item)}, {item.data}")
        return item

    def set(
        self,
        addr: int | tuple[int, int, int, int, int] | Location,
        item: Box,
    ) -> Box:
        """
        Similar to MemSystem.get(), but sets data instead. addr should be of
        the form (channel, rank, bankgroup, bank, local_addr). item should be a
        Box containing the data that should be delivered to the specified
        location. Returns a Box which indicates when the write has completed.
        """
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

        return Box(np.array(item.data), update)

    def local_to_canonical_addr(
        self, location: tuple[int, int, int, int], addr: int
    ) -> int:
        """
        Converts a location (channel, rank, bankgroup, bank) and
        local address to a canonical address. This is a standard
        addressing system between all types of devices which sorts
        the bits of the address by spatial hierarchy.
        """
        return dramsim3.memsys_get_canonical_from_phys(
            self.memsys_ptr, location[0], location[1], location[2], location[3], addr
        )

    def get_config_param(self, id: str) -> int:
        # store values in the python end if possible to avoid string operations
        # / conversions
        if id not in self._params.keys():
            c_id = ctypes.c_char_p(id.encode())
            val: int = dramsim3.memsys_get_config_property(self.memsys_ptr, c_id)
            self._params[id] = val
        return self._params[id]

    def get_active_row(self, channel: int, rank: int, bankgroup: int, bank: int) -> int:
        return dramsim3.memsys_get_active_row(
            self.memsys_ptr, channel, rank, bankgroup, bank
        )

    @property
    def m_gdl_width(self) -> int:
        """The width of the device's GDL in bits."""
        return self.get_config_param("gdl_width")

    @property
    def cycle(self) -> int:
        """
        The current cycle of the memory system (starts at 0 on MemSystem
        creation).
        """
        return dramsim3.memsys_get_cycle(self.memsys_ptr)

    @property
    def num_ranks(self) -> int:
        return dramsim3.memsys_get_ranks(self.memsys_ptr)

    @property
    def num_banks_per_group(self) -> int:
        return dramsim3.memsys_get_banks_per_bankgroup(self.memsys_ptr)

    @property
    def num_banks(self) -> int:
        return (
            self.num_bankgroups_per_rank
            * self.num_banks_per_group
            * self.num_ranks
            * self.num_channels
        )

    @property
    def num_channels(self) -> int:
        return dramsim3.memsys_get_channels(self.memsys_ptr)

    @property
    def num_bankgroups_per_rank(self) -> int:
        return dramsim3.memsys_get_bankgroups_per_rank(self.memsys_ptr)

    @property
    def tck(self) -> np.float32:
        """
        Cycle time of the memory system (measured in ns).
        """
        cf_tck = dramsim3.memsys_get_tck(self.memsys_ptr)
        return np.float32(cf_tck)

    def loc_to_device_addr(
        self, channel: int, rank: int, bankgroup: int, bank: int, local_addr: int
    ) -> int:
        """
        A helper function which converts a passed location (and local address)
        to its corresponding device address.
        """
        return dramsim3.memsys_get_address_from_physical_location(
            self.memsys_ptr, channel, rank, bankgroup, bank, local_addr
        )

    def loc_from_addr(self, addr: int) -> tuple[int, int, int, int, int]:
        """
        A conversion function which accepts a device address and converts it to
        its corresponding physical location (channel, rank, bankgroup, bank,
        local addr).
        """
        channel = ctypes.c_int64(0)
        rank = ctypes.c_int64(0)
        bankgroup = ctypes.c_int64(0)
        bank = ctypes.c_int64(0)
        local_addr = ctypes.c_int64(0)
        dramsim3.memsys_get_physical_location_from_address(
            self.memsys_ptr,
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
        """
        Adds a data structure to the memory device instantaneously. This
        function returns the ID of the added object, which can be used to
        memory map it.
        """
        if isinstance(data_structure, list):
            data_structure = np.array(data_structure, dtype=np.int32)
        self.stored_data_structures.append(DataStructureContainer(data_structure))
        return len(self.stored_data_structures) - 1

    def get_num_data_structures(self) -> int:
        return len(self.stored_data_structures)

    def shift_offset(self, offset: int) -> int:
        """
        A helper function which returns the DRAMsim3-calculated "shift_bits"
        configuration parameter. This parameter is equivalent to the number of
        "don't care" bits at the bottom of each address which arrives at the
        PHY.
        """
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
        """
        A helper function which fetches a GDL-width slice of data at the passed
        location and address.
        """
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
        data: Box,
    ):
        """
        The driving force behind writing data to a bank. Accepts a
        location and bank-local address and writes the passed Box
        to the associated destination.

        This function is un-timed.
        """
        addr: int = self.local_to_canonical_addr(
            (channel, rank, bankgroup, bank), hex_addr
        )
        d, b = self.address_mapper[addr]

        if d == -1:
            raise PimAccessOutOfBoundsError(
                f"PIM access occurred out of bounds at canonical address {addr}"
                + f"(chan:{channel}, rank:{rank}, bg:{bankgroup}, bank:{bank}, addr: {hex_addr})"
            )

        gdl_width_bytes = int(self.m_gdl_width / 8)
        if d == -2:
            # ds = DataStructureContainer(np.zeros(gdl_width_bytes, dtype=np.uint8))
            return
        else:
            ds = self.stored_data_structures[d]

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
        """
        The driving force behind reading data from a bank. Accepts a location
        and bank-local address and reads the data associated with that
        location..

        This function is un-timed.
        """
        addr: int = self.local_to_canonical_addr(
            (channel, rank, bankgroup, bank), hex_addr
        )
        d, b = self.address_mapper[addr]
        if d == -1:
            raise PimAccessOutOfBoundsError(
                f"PIM access occurred out of bounds at canonical address"
                + f" {addr} (chan:{channel}, rank:{rank}, bg:{bankgroup}, "
                + f"bank:{bank}, local: {hex_addr})"
            )

        if length == 0:
            raise Exception(
                f"Tried to read length 0 at addr {hex_addr}\nbank: {bank}"
                + f"\nbankgroup: {bankgroup}\nrank: {rank}\nchannel: {channel}"
                + f"\nstart byte: {b}\ndata index: {d}"
            )

        gdl_width_bytes = int(self.m_gdl_width / 8)
        if d == -2:
            ds = DataStructureContainer(np.array([], dtype=np.uint8))
        else:
            ds = self.stored_data_structures[d]
        result = np.copy(
            np.frombuffer(
                np.frombuffer(ds.data_structure, dtype=np.uint8)[
                    b * gdl_width_bytes : b * gdl_width_bytes + length
                ],
                dtype=dtype,
            )
        )
        if len(result) == 0 and d != -2:
            raise Exception(
                f"Extracted result of length 0: {result} at addr 0d{hex_addr}"
                + f"\nbank: {bank}\nbankgroup: {bankgroup}\nrank: {rank}\nchannel:"
                + f" {channel}\nstart byte: {b}\ndata index: {d}\nstored "
                + f"datastructure:\n{ds}\nwith len: "
                + f"{len(np.frombuffer(ds.data_structure, dtype=np.uint8))}"
            )
        return result

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
        """
        Accepts a channel, rank, bankgroup, bank, and local address (which is
        in GDL chunks) to which the passed data index should be mapped. The
        length and offset parameters are in GDL chunks, not bytes.

        This function is instantaneous and does not carry with it any timing
        data.
        """
        start: int = self.local_to_canonical_addr(
            (channel, rank, bankgroup, bank), hex_addr
        )
        max_addr = self.get_config_param("n_row") * self.get_config_param("n_col")
        # TODO: make this robust enough to handle various core-to-data ratios
        # if hex_addr + length > max_addr:
        #     raise PimMmapOutOfBoundsError(
        #         f"The mmapped data at canonical address {hex(start)}"
        #         + f"within bank (c:{channel}, r:{rank}, bg:{bankgroup}, b:{bank}, "
        #         + f"addr:{hex(hex_addr)}) exceeds limit of a single-bank allocation."
        #         + f"({start + length - max_addr} too many bytes)."
        #     )
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
        """
        Accepts a channel, rank, bankgroup, bank, and local address (which is
        in GDL chunks) which should be unmapped from memory. Note: this does
        *not* delete the mapped objects from host memory. If you wish to do so,
        you should replace MemSystem.stored_data_structures[obj id] with an
        empty array.

        This function is instantaneous and does not carry with it any timing
        data.
        """
        start: int = self.local_to_canonical_addr(
            (channel, rank, bankgroup, bank), hex_addr
        )
        self.address_mapper.remove_mapping(start, start + length)

    def print_config(self) -> None:
        print("Channels:", self.num_channels)
        print("Ranks:", self.num_ranks)
        print("Bankgroups per rank:", self.num_bankgroups_per_rank)
        print("Banks per bankgroup:", self.num_banks_per_group)

    def print_stats(self) -> None:
        dramsim3.memsys_print_stats(self.memsys_ptr)

    def register_callbacks(
        self, read_callback: CallbackType, write_callback: CallbackType
    ) -> None:
        dramsim3.memsys_register_callbacks(
            self.memsys_ptr, read_callback, write_callback
        )

    def tick(self, duration: int = 1, until_event: bool = False) -> None:
        """
        The progression function for the MemSystem class. Advances
        the state by one cycle (default), but can advance by duration cycles
        if duration is passed. Otherwise, the system may be progressed until
        an event has fired if until_event is passed.

        This function does not affect the state of cores, which must be ticked
        independently.
        """
        if until_event:
            while not self.event:
                dramsim3.memsys_tick(self.memsys_ptr)
            self.event = False
            return
        for _ in range(duration):
            dramsim3.memsys_tick(self.memsys_ptr)

    def add_transaction(self, addr: int, is_write: bool, is_pim: bool = False) -> bool:
        """
        Add a memory transaction to the MemSystem by device physical address.

        This function helps handle timing but does not affect state.
        """
        return dramsim3.memsys_add_transaction(
            self.memsys_ptr, addr, is_write, is_pim
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
        """
        Add a memory transaction to the MemSystem by its location and local
        address. Most useful for PIM core read/write operations.

        This function helps handle timing but does not affect state.
        """
        return dramsim3.memsys_add_transaction_to_bank(
            self.memsys_ptr, channel, rank, bankgroup, bank, addr, is_write, is_pim
        )

    def toggle_pim_mode(self) -> None:
        dramsim3.memsys_toggle_mode(self.memsys_ptr)

    def get_pim_mode(self) -> bool:
        """
        Return a boolean representing whether the device is currently in PIM
        mode (True if so).
        """
        return dramsim3.memsys_get_pim_mode(self.memsys_ptr)

    def set_pim_mode(self, mode: bool) -> None:
        dramsim3.memsys_set_pim_mode(self.memsys_ptr, mode)

    def destroy(self) -> None:
        self.destroyed = True
        dramsim3.memsys_destroy(self.memsys_ptr)

    # it's safer to call destroy yourself, but this
    # ideally cleans up your mess if you forget
    def __del__(self) -> None:
        if not self.destroyed:
            self.destroy()
