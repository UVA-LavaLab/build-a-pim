from lib.cores.components.addressing import Swizzler
import pytest
from lib.errors import AddressSwizzleIncompatible


def test_non_overlapping():
    swaps = [0b1101, 0b11010000]
    s = Swizzler(swaps)
    i = 0b1001
    e = 0b10010000
    assert s(i) == e

    i = 0b1101
    e = 0b11010000
    assert s(i) == e

    i = 0b0010
    e = 0b0010
    assert s(i) == e

    i = 0b1111
    e = 0b11010010
    assert s(i) == e

def test_overlapping_fails():
    with pytest.raises(AddressSwizzleIncompatible):
        swaps = [0b1101, 0b1110]
        s = Swizzler(swaps)

def test_different_dimension_fails():
    with pytest.raises(AddressSwizzleIncompatible):
        swaps = [0b1101, 0b1100]
        s = Swizzler(swaps)

def test_wrong_length_fails():
    with pytest.raises(AddressSwizzleIncompatible):
        swaps = [0b1101, 0b1100, 0b0]
        s = Swizzler(swaps)

def test_id_swizzle():
    swaps = [0b1101, 0b11010000, 0b11010000, 0b1101]
    s = Swizzler(swaps)
    i = 0b1101
    assert s(i) == i
