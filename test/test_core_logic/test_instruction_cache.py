from lib.cores.components.instruction_cache import InstructionCache
from lib.cores.instructions import Instruction, OpType
import numpy as np

def setup() -> InstructionCache:
    return InstructionCache()

def gen_prog(length: int) -> list[Instruction]:
    return [Instruction(OpType.NOP) for _ in range(length)]

def test_ins_cache_load_and_fetch():
    size = 32
    ic = setup()
    b, o = ic.load_prog(gen_prog(size))
    assert len(ic._mem) != 0
    assert not o
    assert b == 0

    # check bounds of fetching
    for i in range(size):
        assert ic[i] is not None

    b, o = ic.load_prog(gen_prog(size), size)
    assert not o
    assert b == size
    assert len(ic._mem) == 2

    # check for continuity of fetching
    for i in range(size * 2):
        assert ic[i] is not None

def test_ins_cache_drop():
    size = 32
    ic = setup()
    prog = gen_prog(size)
    b, _ = ic.load_prog(prog)

    r = ic.drop_prog_starting_at(b)
    assert r
    assert len(ic._mem) == 1
    assert ic._mem[0] == []

    b, _ = ic.load_prog(prog)
    assert len(ic._mem) == 1
    assert ic._mem[0] != []

    r = ic.drop_prog_starting_at(b + 1)
    assert r
    assert len(ic._mem) == 1
    assert ic._mem[0] == []

    b, _ = ic.load_prog(prog)
    r = ic.drop_prog_starting_at(b + size)
    assert not r
    assert len(ic._mem) == 1
    assert ic._mem[0] != []

    b, _ = ic.load_prog(prog, addr=size)
    assert len(ic._mem) == 2
    assert ic._mem[1] != []

    r = ic.drop_prog_starting_at(size)
    assert r
    assert len(ic._mem) == 2
    assert ic._mem[1] == []
    assert ic._mem[0] != []
