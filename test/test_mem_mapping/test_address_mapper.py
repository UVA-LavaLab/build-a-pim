from lib.address.address_mapper import AddressMapper
import numpy as np


def setup():
    return AddressMapper()


def test_addr_mapper_singleton_mapping():
    am = setup()
    am.add_mapping(0x0, 0x1, 0, 0)
    am.bake()
    print(am.np_boundaries)
    print(am.np_indices)
    print(am.np_offsets)
    assert am[0] == (0, 0)
    assert am[1] == (-1, 0)


def test_addr_mapper_single_interval_bounds():
    am = setup()
    am.add_mapping(0x0, 0x20, 0, 0)
    am.bake()
    print(am.np_boundaries)
    print(am.np_indices)
    print(am.np_offsets)
    print(am[0])
    for i in range(0, 0x20):
        print(f"testing {hex(i)}")
        assert am[i] == (0, i)
    assert am[0x20] == (-1, 0)


def test_addr_mapper_multi_interval_bounds():
    am = setup()
    am.add_mapping(0x0, 0x20, 0, 0)
    am.add_mapping(0x20, 0x40, 1, 1)
    am.add_mapping(0x40, 0x60, 2, 2)
    am.bake()
    print(am.np_boundaries)
    print(am.np_indices)
    print(am.np_offsets)
    for i in range(0, 0x20):
        assert am[i] == (0, i)
    for i in range(0x20, 0x40):
        assert am[i] == (1, 1 + i - 0x20)
    for i in range(0x40, 0x60):
        assert am[i] == (2, 2 + i - 0x40)
    assert am[0x60] == (-1, 0)


def test_addr_mapper_removal_singleton():
    am = setup()

    def extr_vals(am: AddressMapper):
        return (
            am.boundaries.copy(),
            am.indices.copy(),
            am.offsets,
            np.copy(am.np_boundaries),
            np.copy(am.np_indices),
            np.copy(am.np_offsets),
        )

    am.bake()
    pre_vals = extr_vals(am)
    am.add_mapping(0x0, 0x20, 0, 0)
    am.bake()
    am.remove_mapping(0x0, 0x20)
    am.bake()
    post_vals = extr_vals(am)
    for i, (vpre, vpost) in enumerate(zip(pre_vals, post_vals)):
        print(f"Pre {i}: {vpre}")
    for i, (vpre, vpost) in enumerate(zip(pre_vals, post_vals)):
        print(f"Post {i}: {vpost}")
    for i, (vpre, vpost) in enumerate(zip(pre_vals, post_vals)):
        print(f"Comparing at position {i}")
        print(f"Pre-val: {vpre}")
        print(f"Post-val: {vpost}")
        assert vpre == vpost


def test_addr_mapper_removal_inner_bounds():
    am = setup()

    am.add_mapping(0x0, 0x20, 0, 1)
    am.remove_mapping(0x01, 0x19)
    am.bake()
    assert am.boundaries == [0, 0x1, 0x19, 0x20]
    assert np.all(am.np_boundaries == np.array([0, 0x1, 0x19, 0x20]))
    assert am.indices == [0, -1, 0, -1]
    assert np.all(am.np_indices == np.array([0, -1, 0, -1]))
    assert am.offsets == [1, 0, 1, 0]
    assert np.all(am.np_offsets == np.array([1, 0, 1, 0]))

def test_addr_mapper_removal_around():
    am = setup()

    def extr_vals(am: AddressMapper):
        return (
            am.boundaries.copy(),
            am.indices.copy(),
            am.offsets,
            np.copy(am.np_boundaries),
            np.copy(am.np_indices),
            np.copy(am.np_offsets),
        )

    am.bake()
    pre_vals = extr_vals(am)
    am.add_mapping(0x1, 0x20, 0, 0)
    am.bake()
    am.remove_mapping(0x0, 0x21)
    am.bake()
    post_vals = extr_vals(am)
    for i, (vpre, vpost) in enumerate(zip(pre_vals, post_vals)):
        print(f"Pre {i}: {vpre}")
    for i, (vpre, vpost) in enumerate(zip(pre_vals, post_vals)):
        print(f"Post {i}: {vpost}")
    for i, (vpre, vpost) in enumerate(zip(pre_vals, post_vals)):
        print(f"Comparing at position {i}")
        print(f"Pre-val: {vpre}")
        print(f"Post-val: {vpost}")
        assert vpre == vpost


def test_addr_mapper_removal_multiple():
    am = setup()

    am.bake()
    am.add_mapping(0x0, 0x20, 0, 1)
    am.add_mapping(0x20, 0x40, 1, 2)
    am.remove_mapping(0x5, 0x25)
    am.bake()

    assert am.boundaries == [0x0, 0x5, 0x25, 0x40]
    assert am.indices == [0, -1, 1, -1]
    assert am.offsets == [1, 0, 2, 0]
    assert np.all(am.np_boundaries == np.array([0x0, 0x5, 0x25, 0x40]))
    assert np.all(am.np_indices == np.array([0, -1, 1, -1]))
    assert np.all(am.np_offsets == np.array([1, 0, 2, 0]))


def test_addr_mapper_insert_overlap():
    am = setup()

    am.bake()
    am.add_mapping(0x0, 0x20, 0, 1)
    am.add_mapping(0x15, 0x30, 1, 2)
    am.bake()

    assert am.boundaries == [0x0, 0x15, 0x30]
    assert am.indices == [0, 1, -1]
    assert am.offsets == [1, 2, 0]
    assert np.all(am.np_boundaries == np.array([0x0, 0x15, 0x30]))
    assert np.all(am.np_indices == np.array([0, 1, -1]))
    assert np.all(am.np_offsets == np.array([1, 2, 0]))


def test_addr_mapper_insert_overwrite():
    am = setup()

    am.bake()
    am.add_mapping(0x1, 0x20, 0, 1)
    am.add_mapping(0x0, 0x30, 1, 2)
    am.bake()

    assert am.boundaries == [0x0, 0x30]
    assert am.indices == [1, -1]
    assert am.offsets == [2, 0]
    assert np.all(am.np_boundaries == np.array([0x0, 0x30]))
    assert np.all(am.np_indices == np.array([1, -1]))
    assert np.all(am.np_offsets == np.array([2, 0]))

def test_addr_mapper_insert_spanning_overwrite():
    am = setup()

    am.bake()
    am.add_mapping(0x0, 0x20, 1, 1)
    am.add_mapping(0x20, 0x40, 2, 2)
    am.add_mapping(0x10, 0x30, 3, 3)
    am.bake()

    assert am.boundaries == [0x0, 0x10, 0x30, 0x40]
    assert am.indices == [1, 3, 2, -1]
    assert am.offsets == [1, 3, 2, 0]
    assert np.all(am.np_boundaries == np.array([0x0, 0x10, 0x30, 0x40]))
    assert np.all(am.np_indices == np.array([1, 3, 2, -1]))
    assert np.all(am.np_offsets == np.array([1, 3, 2, 0]))

def test_addr_mapper_insert_far_apart():
    am = setup()

    am.bake()
    am.add_mapping(0x0, 0x20, 1, 1)
    am.add_mapping(0x100, 0x110, 2, 2)
    am.bake()

    assert am.boundaries == [0x0, 0x20, 0x100, 0x110]
    assert am.indices == [1, -1, 2, -1]
    assert am.offsets == [1, 0, 2, 0]
    assert np.all(am.np_boundaries == np.array([0x0, 0x20, 0x100, 0x110]))
    assert np.all(am.np_indices == np.array([1, -1, 2, -1]))
    assert np.all(am.np_offsets == np.array([1, 0, 2, 0]))

def test_addr_mapper_empty_insertion():
    am = setup()

    am.bake()
    am.add_mapping(0x0, 0, 1, 1)
    am.bake()

    assert am.boundaries == [0]
    assert am.indices == [-1]
    assert am.offsets == [0]
    assert np.all(am.np_boundaries == np.array([0]))
    assert np.all(am.np_indices == np.array([-1]))
    assert np.all(am.np_offsets == np.array([0]))

def test_addr_mapper_redundant_removal():
    am = setup()

    am.bake()
    am.add_mapping(0x0, 0x10, 1, 1)
    am.add_mapping(0x20, 0x30, 2, 2)
    # tests both removal edge cases in one go
    am.remove_mapping(0x10, 0x20)
    am.remove_mapping(0x11, 0x19)
    am.bake()

    assert am.boundaries == [0x0, 0x10, 0x20, 0x30]
    assert am.indices == [1, -1, 2, -1]
    assert am.offsets == [1, 0, 2, 0]
    assert np.all(am.np_boundaries == np.array([0x0, 0x10, 0x20, 0x30]))
    assert np.all(am.np_indices == np.array([1, -1, 2, -1]))
    assert np.all(am.np_offsets == np.array([1, 0, 2, 0]))
