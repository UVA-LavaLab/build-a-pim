from enum import Enum
from lib.types import PimRegType
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


class Command:
    """
    The Pim Command class. This class should encapsulate the basic
    function of a PIM command and should be easily extensible.
    """

    def __init__(
        self,
        type: CommandType = CommandType.NOP,
        operand_1_range: tuple[int, int] = (0, 0),
        operand_2_range: tuple[int, int] = (0, 0),
        dst_reg: PimRegType[Any] | None = None,
    ):
        self.cmdtype: CommandType = type
        self.operand_1_range: tuple[int, int] = operand_1_range
        self.operand_2_range: tuple[int, int] = operand_2_range
        if dst_reg is not None:
            self.dst_reg: PimRegType[Any] = dst_reg
