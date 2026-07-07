from ast import match_case
from enum import Enum
from lib.memsys import MemSystem
from lib.cores.components.base import BaseCore
from lib.cores.ideal_lobsta_simd import Core as LobstaCore
from lib.errors import AllocationStrategyNotSupportedError
import numpy.typing as npt
import numpy as np
import math


class AllocationStrategy(Enum):
    """
    An enum of allocation strategies. The width of the allocation strategy is
    the amount of data from the original contiguous buffer which is
    contiguously stored in the core's local bank. In the case of MAX_WIDTH,
    this means that all data is stored in the maximum possible contiguous width
    on each bank. "XB_YC" is the ratio of cores banks to cores.
    """

    ROUND_ROBIN_ROW_WIDTH_1B_1C = 0
    ROUND_ROBIN_GDL_WIDTH_1B_1C = 1
    ROUND_ROBIN_MAX_WIDTH_1B_1C = 2
    LOBSTA_CONTIGUOUS_INSIDE_SUB_ROWS = 3


def pim_device_place_data(
    mem: MemSystem,
    cores: list[BaseCore],
    object: npt.NDArray[np.generic],
    addr: int,
    strategy: AllocationStrategy = AllocationStrategy.ROUND_ROBIN_MAX_WIDTH_1B_1C,
    core_location_offset: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> tuple[int, tuple[int, int]]:
    """
    Accepts a MemSystem object, a list of cores, a numpy array, an address and
    an optional allocation strategy.

    Returns a tuple containing the ID of the inserted data structure and a
    tuple containing the first local address allocated and the address after
    the final allocation.

    Ranges which have been allocated but are not used are mapped to the ID -2
    in the device's address mapper.
    """
    data_ind = mem.add_data_structure(np.copy(object))

    match strategy:
        # FIXME: this allocation strategy will also result in large holes of
        # unusable memory (up to 15 words) in cases where the allocation maps a
        # passed array which is not uniformly distributed
        # FIXME: this causes an out of bounds read hazard when banks cannot
        # evenly divide GDL chunks!!
        case AllocationStrategy.ROUND_ROBIN_MAX_WIDTH_1B_1C:
            obj_bytes: npt.NDArray[np.uint8] = np.frombuffer(object, dtype=np.uint8)
            gdl_bytes: int = int(mem.m_gdl_width / 8)
            # compute length of the input in GDL slices
            len_bytes: int = int(math.ceil(len(obj_bytes) / gdl_bytes)) * gdl_bytes

            len_chunks: int = int(len_bytes / gdl_bytes)
            # this is an over-estimate, needs to be bounds-checked later
            chunks_per_core: int = int(math.ceil(len_chunks / len(cores)))
            cumulative_offset: int = 0

            for core in cores:
                chan, rank, bg, bank = core.location
                chan += core_location_offset[0]
                rank += core_location_offset[1]
                bg += core_location_offset[2]
                bank += core_location_offset[3]
                # we want to map to the same address in every bank
                # the length of the data mapped should be equal to (len_chunks / num cores)
                # then the offset should be the cumulative length of already mapped GDL chunks
                length = min(len_chunks - cumulative_offset, chunks_per_core)
                mem.mmap(
                    chan,
                    rank,
                    bg,
                    bank,
                    addr,
                    data_index=data_ind,
                    length=length,
                    offset=cumulative_offset,
                )
                cumulative_offset += length
                if length < chunks_per_core:
                    mem.mmap(
                        chan,
                        rank,
                        bg,
                        bank,
                        addr + length,
                        data_index=-2,
                        length=chunks_per_core - length,
                        offset=0,
                    )
            return data_ind, (addr, addr + chunks_per_core)
        case AllocationStrategy.LOBSTA_SPMV_DENSE:
            obj_bytes: npt.NDArray[np.uint8] = np.frombuffer(object, dtype=np.uint8)
            gdl_bytes: int = int(mem.m_gdl_width / 8)
            # compute length of the input in GDL slices
            len_bytes: int = int(math.ceil(len(obj_bytes) / gdl_bytes)) * gdl_bytes

            len_chunks: int = int(len_bytes / gdl_bytes)
            # this is an over-estimate, needs to be bounds-checked later
            chunks_per_core: int = int(math.ceil(len_chunks / len(cores)))
            cumulative_offset: int = 0

            for core in cores:
                chan, rank, bg, bank = core.location
                chan += core_location_offset[0]
                rank += core_location_offset[1]
                bg += core_location_offset[2]
                bank += core_location_offset[3]
                # we want to map to the same address in every bank
                # the length of the data mapped should be equal to (len_chunks / num cores)
                # then the offset should be the cumulative length of already mapped GDL chunks
                length = min(len_chunks - cumulative_offset, chunks_per_core)
                mem.mmap(
                    chan,
                    rank,
                    bg,
                    bank,
                    addr,
                    data_index=data_ind,
                    length=length,
                    offset=cumulative_offset,
                )
                cumulative_offset += length
                if length < chunks_per_core:
                    mem.mmap(
                        chan,
                        rank,
                        bg,
                        bank,
                        addr + length,
                        data_index=-2,
                        length=chunks_per_core - length,
                        offset=0,
                    )
            return data_ind, (addr, addr + chunks_per_core)
        case _:
            raise AllocationStrategyNotSupportedError(
                f"Allocation strategy {strategy} not supported."
            )

    raise NotImplementedError("Not implemented return of addr in pim_device_place_data")
    return -1, 0


def lobsta_place_data(
    mem: MemSystem,
    cores: list[LobstaCore],
    object: npt.NDArray[np.generic],
    addr: int,
    strategy: AllocationStrategy = AllocationStrategy.LOBSTA_CONTIGUOUS_INSIDE_SUB_ROWS,
    sub_row_indices: list[int] | None = None,
) -> tuple[int, tuple[int, int]]:
    """
    Accepts a MemSystem object, a list of cores, a numpy array, an address and
    an optional allocation strategy.

    Returns a tuple containing the ID of the inserted data structure and a
    tuple containing the first local address allocated and the address after
    the final allocation.

    Ranges which have been allocated but are not used are mapped to the ID -2
    in the device's address mapper.
    """
    data_ind = mem.add_data_structure(np.copy(object))
    sub_rows: list[int] = (
        list(range(cores[0].sub_rows)) if sub_row_indices is None else sub_row_indices
    )

    match strategy:
        # FIXME: this allocation strategy will also result in large holes of
        # unusable memory (up to 15 words) in cases where the allocation maps a
        # passed array which is not uniformly distributed
        # FIXME: this causes an out of bounds read hazard when banks cannot
        # evenly divide GDL chunks!!
        case AllocationStrategy.LOBSTA_CONTIGUOUS_INSIDE_SUB_ROWS:
            obj_bytes: npt.NDArray[np.uint8] = np.frombuffer(object, dtype=np.uint8)
            gdl_bytes: int = int(mem.m_gdl_width / 8)
            # compute length of the input in GDL slices
            len_bytes: int = int(math.ceil(len(obj_bytes) / gdl_bytes)) * gdl_bytes

            len_chunks: int = int(len_bytes / gdl_bytes)
            # this is an over-estimate, needs to be bounds-checked later
            chunks_per_core: int = int(math.ceil(len_chunks / len(cores)))
            # quickly calculate the number of addressable columns
            max_chunks_per_sub_row: int = cores[0].p_mem().get_config_param("co_mask") + 1
            cumulative_offset: int = 0

            for core in cores:
                chan, rank, bg, bank = core.location
                sub_row_offset: int = 0
                for _ in range(0, chunks_per_core, max_chunks_per_sub_row * len(sub_rows)):
                    for sub_row in sub_rows:
                        length = min(
                            len_chunks - cumulative_offset,
                            chunks_per_core,
                            max_chunks_per_sub_row,
                        )
                        if length <= 0:
                            break
                        mem.mmap(
                            chan,
                            rank,
                            bg,
                            bank + sub_row,
                            addr + sub_row_offset,
                            data_index=data_ind,
                            length=length,
                            offset=cumulative_offset,
                        )
                        cumulative_offset += length
                        # pads the end of the allocation
                        # NOTE: this null-padded allocation strategy is much
                        # better on LoBSTA because it wastes less space
                        if length < max_chunks_per_sub_row:
                            mem.mmap(
                                chan,
                                rank,
                                bg,
                                bank + sub_row,
                                addr + length + sub_row_offset,
                                data_index=-2,
                                length=max_chunks_per_sub_row - length,
                                offset=0,
                            )
                    sub_row_offset += max_chunks_per_sub_row
            return data_ind, (addr, addr + chunks_per_core)
        case _:
            raise AllocationStrategyNotSupportedError(
                f"Allocation strategy {strategy} not supported."
            )
