from abc import ABC, abstractmethod
from collections import deque
from lib.monad import DataWrapper, Ptr
from lib.memsys import MemSystem
from lib.cores.instructions import OpType, Instruction
from lib.controller.commands import Command, CommandType
from lib.controller.response import Response
from typing import Any
import numpy.typing as npt
import numpy as np


class BaseCore(ABC):
    supported_cmds: list[CommandType] = []
    timings: dict[OpType, int] = {}

    def __init__(
        self,
        location: tuple[int, int, int, int],
        p_mem: Ptr[MemSystem],
        registers: list[str] | None = None,
        vec_registers: list[str] | None = None,
        tCK: float = 5.0,
    ):
        self.gdl: DataWrapper = DataWrapper([], None)
        self.cycle: int = 0

        self.channel: int = location[0]
        self.rank: int = location[1]
        self.bankgroup: int = location[2]
        self.bank: int = location[3]
        self.p_mem: Ptr[MemSystem] = p_mem
        self.instruction_queue: deque[Instruction] = deque()
        self.tCK: np.float32 = np.float32(tCK)

        self.reg: dict[str, DataWrapper] = {}
        if registers is None:
            self.registers: list[str] = ["regA", "regB", "regC"]
        else:
            self.registers = registers

        for r in self.registers:
            setattr(self, r, 0)
            self.__class__.__annotations__[r] = int | float

        if vec_registers is None:
            self.vec_registers: list[str] = ["reg_vA", "reg_vB", "reg_vC"]
        else:
            self.vec_registers = vec_registers

        for r in self.vec_registers:
            setattr(self, r, DataWrapper([]))
            self.__class__.__annotations__[r] = DataWrapper | None

    @property
    def location(self):
        return (self.channel, self.rank, self.bankgroup, self.bank)

    @property
    def isa(self):
        return self.timings.keys()

    @abstractmethod
    def ins_queue_handler(self):
        pass

    @abstractmethod
    def cmd_handler(self, cmd: Command | None):
        pass

    @abstractmethod
    def tick(self, cmd: Command | None = None) -> Response | None:
        self.cmd_handler(cmd)
        self.ins_queue_handler()
        self.cycle += 1

    @abstractmethod
    def instruction_side_effect_callback(self, ins: Instruction):
        pass

    def location_plus_addr(self, addr: int = 0) -> tuple[int, int, int, int, int]:
        return (self.channel, self.rank, self.bankgroup, self.bank, addr)

    def get_reg(self, reg: str) -> Any:
        rval: DataWrapper | None = getattr(self, reg)
        if rval is None:
            rval = DataWrapper([])
        return rval

    def set_reg(self, reg: str, val: DataWrapper | Any):
        if not isinstance(val, DataWrapper):
            assert reg in self.registers
        else:
            assert reg in self.vec_registers
        setattr(self, reg, val)

    def call_start_setter(self, ins: Instruction):
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
                    ins.in_reg1 != "" or ins.in_reg2 != "",
                    "Undefined behavior: one or more input registers are set for READ instruction.",
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
                    ins.in_reg2 != "",
                    "Undefined behavior: Secondary input register (in_reg2) behavior not defined.",
                )

                def scb():
                    dst: DataWrapper = (
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
                ins.is_done = lambda: ins.data.is_ready()
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

    # Enforces class variable declaration requirements
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
