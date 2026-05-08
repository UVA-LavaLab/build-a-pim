from lib.controller.commands import Command, CommandType
from typing import Any
import numpy.typing as npt
import numpy as np


class DeviceCommand:
    def __init__(
        self,
        cmdtype: CommandType,
        op1_id: int | None = None,
        op2_id: int | None = None,
        dst_id: int | None = None,
        scalar: Any = None,
        dtype: npt.DTypeLike = np.int32,
    ):
        self.cmdtype: CommandType = cmdtype
        self.op1_id: int = -1 if op1_id is None else op1_id
        self.op2_id: int = -1 if op2_id is None else op2_id
        self.dst_id: int = -1 if dst_id is None else dst_id
        self.scalar: Any = scalar
        self.dtype: npt.DTypeLike = dtype

    def to_command(self, mapping: dict[int, tuple[int, int]]) -> Command:
        return Command(
            type=self.cmdtype,
            operand_1=mapping[self.op1_id][0],
            operand_2=mapping[self.op1_id][1],
            operand_3=mapping[self.op2_id][0],
            operand_4=mapping[self.op2_id][1],
            dst_1=mapping[self.dst_id][0],
            dst_2=mapping[self.dst_id][1],
            scalar=self.scalar,
            dtype=self.dtype,
        )
