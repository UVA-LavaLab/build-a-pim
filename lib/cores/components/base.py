from abc import ABC, abstractmethod
from collections import deque
from lib.containers import Box, Ptr
from lib.memsys import MemSystem
from lib.cores.instructions import OpType, Instruction
from lib.controller.commands import Command, CommandType
from lib.controller.response import Response
from typing import Any, Callable
import numpy.typing as npt
import numpy as np
from math import ceil


def genRegSet(num_registers: int, vector: bool = False):
    step_size: int = ord("Z") - ord("A") + 1
    prefix: str = "vr" if vector else "r"
    def name(index: int) -> str:
        chars: list[str] = []
        index += 1
        while index > 0:
            index, rem = divmod(index - 1, step_size)
            chars.append(chr(ord("A") + rem))
        return "".join(reversed(chars))

    
    return [prefix + name(i) for i in range(num_registers)]


class BaseCore(ABC):
    """
    The base class for all cores. This class is designed to enforce a certain
    structure on arbitrary cores. It separates registers and vector registers
    as well as requires each class to have an associated physical location.

    If you want to have a pipelined processor, you must implement the pipeline
    at the next higher level of abstraction, since including it in this class
    would cause a cyclic dependency.

    There is no established register convention for BaseCore (or any core in
    Build-A-PIM), as conventions are implementation-specific.
    """

    supported_cmds: list[CommandType] = []
    timings: dict[OpType, int] = {}

    def __init__(
        self,
        location: tuple[int, int, int, int],
        p_mem: Ptr[MemSystem],
        num_registers: int = 26,
        num_vec_registers: int = 26,
        registers: list[str] | None = None,
        vec_registers: list[str] | None = None,
        tCK: float = 5.0,
    ):
        """The location parameter must be of the form (channel, rank, bankgroup, bank)."""
        self.gdl: Box = Box(np.array([]), None)
        self.cycle: int = 0

        self.channel: int = location[0]
        self.rank: int = location[1]
        self.bankgroup: int = location[2]
        self.bank: int = location[3]
        self.p_mem: Ptr[MemSystem] = p_mem
        self.instruction_queue: deque[Instruction] = deque()
        self.tCK: np.float32 = np.float32(tCK)

        self.reg: dict[str, Box] = {}
        if registers is None:
            self.registers: list[str] = genRegSet(num_registers)
        else:
            self.registers = registers

        for r in self.registers:
            setattr(self, r, 0)
            self.__class__.__annotations__[r] = int | float

        if vec_registers is None:
            self.vec_registers: list[str] = genRegSet(num_vec_registers, vector=True)
        else:
            self.vec_registers = vec_registers

        for r in self.vec_registers:
            setattr(self, r, Box(np.array([])))
            self.__class__.__annotations__[r] = Box | None

    @property
    def location(self):
        """The location of the core in terms of (channel, rank, bankgroup, bank)"""
        return (self.channel, self.rank, self.bankgroup, self.bank)

    @property
    def isa(self):
        """A list of the instructions supported by the called core."""
        return self.timings.keys()

    @abstractmethod
    def ins_queue_handler(self):
        """This method should implement the logic of the instruction queue / cache."""
        pass

    @abstractmethod
    def cmd_handler(self, cmd: Command | None):
        """This method should define how cores parse commands."""
        pass

    @abstractmethod
    def tick(self, cmd: Command | None = None) -> Response | None:
        """
        Progresses the core state by one cycle, updating state accordingly (at
        as much detail as possible for an abstract class). Be sure to call this
        method whenever designing the tick method for a subclass.
        """
        self.cmd_handler(cmd)
        self.ins_queue_handler()
        self.cycle += 1

    @abstractmethod
    def instruction_side_effect_callback(self, ins: Instruction):
        """This method should define any side effects which occur as the result
        of an instruction leaving the pipeline or finishing execution."""
        pass

    def location_plus_addr(self, addr: int = 0) -> tuple[int, int, int, int, int]:
        """Returns the location appended with the passed address. Used for serialization"""
        return (self.channel, self.rank, self.bankgroup, self.bank, addr)

    def get_reg(self, reg: str) -> Any:
        """Gets the state of the specified register."""
        rval: Any = getattr(self, reg)
        if rval is None:
            rval = Box(np.array([]))
        return rval

    def set_reg(self, reg: str, val: Box | Any):
        """Sets the state of the passed register to the passed value. Any state
        found in vector registers must be contained in a Box."""
        if not isinstance(val, Box):
            assert reg in self.registers
        else:
            assert reg in self.vec_registers
        setattr(self, reg, val)

    def call_start_setter(self, ins: Instruction):
        """Mutates the passed instructions such that they will behave as
        expected in the pipeline."""

        def ifail(cond: bool, errmsg: str):
            if cond:
                raise Exception(errmsg)

        match ins.operation:
            case OpType.READ:
                ifail(
                    ins.addr <= -1,
                    "No address supplied for instruction READ.",
                )
                ifail(
                    ins.in_reg2 != "",
                    "Undefined behavior: secondary input register (in_reg2) set for READ instruction.",
                )

                def scb():
                    ins.data = self.p_mem().get(
                        (
                            self.channel,
                            self.rank,
                            self.bankgroup,
                            self.bank,
                            # makes the interpreter not freak out
                            int(ins.addr),
                        )
                    )

                ins.start_cb = scb
                ins.set_is_done(lambda: ins.data.is_ready())

            case OpType.WRITE:
                ifail(
                    ins.addr <= -1,
                    "No address supplied for instruction WRITE.",
                )
                ifail(
                    ins.in_reg2 != "" and ins.in_reg2 not in self.registers,
                    "Undefined behavior: Secondary input register (in_reg2) is not a scalar"
                    + "register; behavior not defined.",
                )

                def scb():
                    dst: Box = (
                        self.get_reg(ins.in_reg1) if ins.in_reg1 != "" else self.gdl
                    )

                    ins.data = self.p_mem().set(
                        (
                            self.channel,
                            self.rank,
                            self.bankgroup,
                            self.bank,
                            int(ins.addr),
                        ),
                        dst,
                    )

                ins.start_cb = scb
                ins.set_is_done(lambda: ins.data.is_ready())
            case _:
                pass

    def add_instruction(
        self,
        op: OpType,
        in_reg1: str | None = None,
        in_reg2: str | None = None,
        dst: str | None = None,
        addr: int | None = None,
        dtype: npt.DTypeLike = np.int32,
    ):
        """Adds the passed instruction to the instruction queue
        defined here. If your core class does not use this queue for
        instruction behavior management, do not use this method."""
        self.instruction_queue.append(
            Instruction(
                op,
                in_reg1=in_reg1,
                in_reg2=in_reg2,
                dst=dst,
                addr=addr,
                completion_time=self.timings[op],
                dtype=dtype,
            )
        )

    # Enforces method declaration requirements for subclasses (must define both
    # tick and ins_queue_handler).
    @classmethod
    def __subclasshook__(cls, C):
        if cls is BaseCore:
            keys = ["tick", "ins_queue_handler"]
            if all(any(key in B.__dict__ for B in C.__mro__) for key in keys):
                return True
        return NotImplemented

    # Enforces class variable declaration requirements
    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        def maybe_fail(var, vartype, purpose):
            if var not in cls.__dict__:
                raise TypeError(
                    f"Subclass {cls.__name__} must define class variable {var}: {vartype}.\n\tPurpose: {purpose}"
                )

        maybe_fail(
            "supported_cmds",
            "list[CommandType]",
            "Advertises functionality to the memory controller.",
        )
        maybe_fail(
            "timings",
            "dict[OpType, int]",
            "Defines the number of cycles an instruction must spend in the execute stage.",
        )
