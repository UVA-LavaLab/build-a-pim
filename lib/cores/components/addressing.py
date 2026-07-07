from lib.errors import AddressSwizzleIncompatible
from typing import Callable


class Swizzler:
    """
    A simple address swizzling class which swaps bits according to the masks
    passed as parameters. Accepts a list of integers, which will swapped
    pairwise. These integers must be bitwise disjoint (pairwise) and equivalent
    bit count (pairwise). 

    Any passed swap_functions will be called upon the resulting address to
    allow for full swizzling compatibility.

    For example, [0b00001101, 0b11010000] is valid, but [0b00010101,
    0b11010000] is not because the masks overlap.
    """

    def __init__(self, swaps: list[int] | None = None, swap_functions: list[Callable[[int], int]] | None = None):
        swaps: list[int] = [] if swaps is None else swaps
        self.swap_functions: list[Callable[[int], int]] = swap_functions if swap_functions is not None else []

        if len(swaps) % 2 != 0:
            raise AddressSwizzleIncompatible(
                f"Length of passed swaps list {len(swaps)} not viable. (Must be even.)"
            )

        self.swaps: list[tuple[int, int, list[int], list[int]]] = []
        for x, y in zip(swaps[::2], swaps[1::2]):
            if x.bit_count() != y.bit_count():
                raise AddressSwizzleIncompatible(
                    f"Lengths of masks are incompatible. (bit count {x.bit_count()}"
                    + f"not compatible with bit count {y.bit_count()})"
                )

            if x & y != 0:
                raise AddressSwizzleIncompatible(
                    f"Masks {x} and {y} are incompatible because they share overlapping bits."
                )
            x_pos: list[int] = [1 << i for i in range(x.bit_length()) if x & (1 << i)]
            y_pos: list[int] = [1 << i for i in range(y.bit_length()) if y & (1 << i)]
            self.swaps.append((x, y, x_pos, y_pos))

    def __call__(self, addr: int):
        for swap in self.swaps:
            original: int = addr
            # zero out bits that need to be swizzled
            addr = original & ~(swap[0] | swap[1])

            for x_bit, y_bit in zip(swap[2], swap[3]):
                if original & x_bit:
                    addr |= y_bit
                if original & y_bit:
                    addr |= x_bit

        for swap_f in self.swap_functions:
            addr = swap_f(addr)

        return addr
