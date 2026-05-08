import numpy as np
import numpy.typing as npt
from bisect import bisect_left, bisect_right
from lib.errors import AddressMappingNotAscendingError


class AddressMapper:
    """
    Maps intervals to their corresponding IDs within the PIM device.
    """
    def __init__(self):
        self.boundaries: list[int] = [0]
        self.indices: list[int] = [-1]
        self.offsets: list[int] = [0]
        self.dirty: bool = True
        self.np_boundaries: npt.NDArray[np.uint64] = np.array([])
        self.np_indices: npt.NDArray[np.int64] = np.array([])
        self.np_offsets: npt.NDArray[np.uint64] = np.array([])

    def add_mapping(self, start: int, end: int, index: int, offset: int):
        """
        Add a mapping between start and end which corresponds to an access to
        {index} which starts at the {offset}'th byte of the underlying data
        structure.
        """
        if end < start:
            raise AddressMappingNotAscendingError(
                f"Address Mapping is not ascending (start {start} > end {end})"
            )
        idx_end = bisect_right(self.boundaries, end)
        idx_start = bisect_left(self.boundaries, start)
        after_val = self.indices[idx_end - 1] if idx_end > 0 else -1
        after_off = self.offsets[idx_end - 1] if idx_end > 0 else 0

        del self.boundaries[idx_start:idx_end]
        del self.indices[idx_start:idx_end]
        del self.offsets[idx_start:idx_end]

        self.insert_boundary(start, index, offset)
        self.insert_boundary(end, after_val, offset=after_off)

        # now, clean up the redundant address mappings
        i = 1
        while i < len(self.indices):
            if (
                self.indices[i] == self.indices[i - 1]
                and self.offsets[i] == self.offsets[i - 1]
            ):
                del self.boundaries[i]
                del self.indices[i]
                del self.offsets[i]
            else:
                i += 1

        self.dirty = True

    def remove_mapping(self, start: int, end: int):
        """
        Removes all mappings between start and end. Functionally equivalent to
        mapping the passed range to the NULL mapping (-1).
        """
        self.add_mapping(start, end, -1, 0)

    def insert_boundary(self, addr: int, val: int, offset: int):
        idx = bisect_left(self.boundaries, addr)
        if idx < len(self.boundaries) and self.boundaries[idx] == addr:
            self.indices[idx] = val
            self.offsets[idx] = offset
        else:
            self.boundaries.insert(idx, addr)
            self.indices.insert(idx, val)
            self.offsets.insert(idx, offset)

    def contains_mapping(self, start: int, end: int) -> bool:
        """
        Returns True when the passed range contains some mapped data within it.
        """
        idx_end = bisect_left(self.boundaries, end) - 1
        idx_start = bisect_right(self.boundaries, start) - 1

        return any(self.indices[i] != -1 for i in range(idx_start, idx_end + 1))

    def get_end_of_range(self, start: int) -> int:
        """
        Return the end of the mapping range in which start is located.
        """
        idx_end = bisect_right(self.boundaries, start)
        return self.boundaries[idx_end]

    def bake(self):
        """
        "Bake" the boundaries stored in Python lists into numpy arrays after
        they are no longer likely to change so we can search them more quickly.
        """
        if self.dirty:
            self.np_boundaries = np.array(self.boundaries, dtype=np.uint64)
            self.np_indices = np.array(self.indices, dtype=np.int64)
            self.np_offsets = np.array(self.offsets, dtype=np.uint64)
            self.dirty = False

    def __getitem__(self, addr: int) -> tuple[int, int]:
        """
        Returns the ID of the object mapped to the passed address and the byte
        offset of the access corresponding to the passed address.
        """
        self.bake()
        bin_idx: int = int(np.searchsorted(self.np_boundaries, addr, side="right"))
        offset: int = int(addr - self.np_boundaries[bin_idx - 1])

        if np.isscalar(addr):
            return (
                (
                    int(self.np_indices[bin_idx - 1]),
                    int(self.np_offsets[bin_idx - 1] + offset),
                )
                if bin_idx >= 0
                else (-1, 0)
            )
