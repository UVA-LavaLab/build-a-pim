from lib.cores.instructions import Instruction, OpType
from lib.address.address_mapper import AddressMapper
from lib.errors import PimAccessOutOfBoundsError


class InstructionCache:
    """
    This class is designed to simulate PC-based accesses to the instruction
    cache. It can be accessed using bracket notation or using the fetch()
    member function. By default, it assumes 32-bit instructions and a 32 KB
    size, inspired by UPMEM's hardware properties.

    This class makes several assumptions, which we intend to relax in a future
    iteration:
        1. All instructions are the same size (4 bytes)
        2. Instruction sizes divide the size of the cache evenly
        3. The instruction cache is addressed at the instruction granularity
    """

    def __init__(self, instruction_size_bits: int = 32, size_bytes: int = 2**15):
        self._mem: list[list[Instruction]] = []
        self._am: AddressMapper = AddressMapper()
        self.size: int = int((size_bytes * 8) / instruction_size_bits)
        self.last_program_addr: int = 0

    def load_prog(self, prog: list[Instruction], addr: int = -1) -> tuple[int, bool]:
        """
        A helper function which loads microprogram data to the instruction
        cache. An address at which the program should be loaded can optionally
        be supplied. It returns a tuple containing the first address of the
        loaded program and a bool indicating whether an instruction was
        overwritten during this program loading process.

        If no address is provided, the program will be loaded at address 0x0.
        """
        start_addr: int = addr if addr > -1 else self.last_program_addr
        if start_addr + len(prog) > self.size:
            raise PimAccessOutOfBoundsError(
                f"Failed to load program. Reason: length of program ({len(prog)})"
                + f" maps to range outside of instruction cache bounds."
                + f"\n{start_addr}:{start_addr + len(prog)} / {self.size}"
            )

        # offset all of the jump instructions by the address to which the base
        # address is mapped
        for ins in prog:
            if ins.is_jump():
                ins.addr += start_addr

        # linearly scan for a free slot in self._mem
        idx = -1
        for i, l in enumerate(self._mem):
            if l == []:
                idx = i

        if idx != -1:
            self._mem[idx] = prog
        else:
            idx = len(self._mem)
            self._mem.append(prog)

        overwritten: bool = self._am.contains_mapping(
            start_addr, start_addr + len(prog)
        )
        self._am.add_mapping(start_addr, start_addr + len(prog), idx, 0)
        self.last_program_addr = start_addr + len(prog)

        return (start_addr, overwritten)

    def drop_prog_starting_at(self, addr: int) -> bool:
        """
        A helper function which removes a program from the instruction cache.
        Any program which contains an instruction at the passed address will be
        dropped.
        """
        idx, _ = self._am[addr]
        if idx != -1:
            self._mem[idx] = []
            self._am.remove_mapping(addr, self._am.get_end_of_range(addr))
        return idx != -1

    def fetch(self, pc: int) -> Instruction | None:
        """
        A helper function which fetches the instruction at the specified
        program counter.
        """
        if pc > self.size:
            raise PimAccessOutOfBoundsError(
                f"Failed to load program. Reason: PC access out of bounds:"
                + f" {pc} / {self.size} (PC / max addr)."
            )
        idx, offset = self._am[pc]
        return self._mem[idx][offset].clone() if idx != -1 else None

    def __getitem__(self, pc: int) -> Instruction | None:
        return self.fetch(pc)
