from enum import Enum
from lib.types import PimRegType
from lib.errors import MemCmdMalformedError
from typing import Any


class CommandType(Enum):
    """
    A canonical list of available PIM commands.

    A list of supported commands (selected from here or some class
    which extends CommandType) should be advertised from the device
    class to the controller class.

    This list of commands is partially influenced by the PIMeval API.
    """

    NOP = -1
    SWITCH_MODE_PIM = 0
    SWITCH_MODE_MEM = 1
    PIM_ADD = 2
    PIM_SUB = 3
    PIM_MUL = 4
    PIM_DIV = 5
    PIM_ABS = 6
    PIM_NOT = 7
    PIM_AND = 8
    PIM_OR = 9
    PIM_XOR = 10
    PIM_XNOR = 11
    PIM_MIN = 12
    PIM_MAX = 13
    # broadcast a set of bytes to cores
    PIM_BROADCAST = 14
    # sum reduce
    PIM_RED_SUM = 15
    PIM_RED_MUL = 16
    PIM_MALLOC = 17
    PIM_MAC = 19
    PIM_SCALED_ADD = 19
    PIM_POPCOUNT = 20
    PIM_REG_TO_HOST = 21
    PIM_START_EXECUTION = 22
    PIM_END_EXECUTION = 23
    PIM_START_PROGRAM_LOAD = 24
    PIM_END_PROGRAM_LOAD = 25
    MEM_READ = 26
    MEM_WRITE = 27
    PIM_FREE = 28

    def is_mem(self):
        return self == CommandType.MEM_READ or self == CommandType.MEM_WRITE

    def is_malloc(self):
        return self == CommandType.PIM_MALLOC

    def is_free(self):
        return self == CommandType.PIM_FREE


class Command:
    """
    The Pim Command class. This class should encapsulate the basic
    function of a PIM command and should be easily extensible.
    """

    def __init__(
        self,
        type: CommandType = CommandType.NOP,
        operand_1: int = 0,
        operand_2: int = 0,
        operand_3: int = 0,
        operand_4: int = 0,
        scalar: Any = None,
        dst_reg: PimRegType[Any] | None = None,
    ):
        self.cmdtype: CommandType = type

        self.range_1: tuple[int, int] = (operand_1, operand_2)
        self.range_2: tuple[int, int] = (operand_3, operand_4)

        self.address = operand_1 # used in mem transactions

        if dst_reg is not None:
            self.dst_reg: PimRegType[Any] = dst_reg

    @property
    def addr(self):
        if not self.cmdtype.is_mem():
            raise MemCmdMalformedError(
                "Only memory commands are compatible with the 'addr' property."
            )
        return self.address
