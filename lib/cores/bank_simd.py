from lib.errors import (
    PimCmdNotSupportedError,
    PimCmdNotImplementedError,
    PimInstructionMalformedError,
)
from lib.memsys import MemSystem
from lib.cores.instructions import Instruction, OpType
from lib.cores.components.base import BaseCore
from lib.monad import DataWrapper, Ptr
from lib.controller.commands import CommandType, Command
from lib.cores.components.pipeline import (
    Stage,
    Pipeline,
    mkDefaultStages,
)
from lib.cores.components.functional import (
    dtype_min,
    dtype_max,
    map_scalar_vec,
    map_vec,
    fold_vec,
    red_kernel,
    vec_scalar_kernel,
    vec_vec_kernel,
)
from typing import override, Callable
import numpy as np
import math


class Core(BaseCore):
    supported_cmds: list[CommandType] = [
        CommandType.PIM_ADD,
        CommandType.PIM_SUB,
        CommandType.PIM_DIV,
        CommandType.PIM_MUL,
        CommandType.PIM_ABS,
        CommandType.PIM_RED_SUM,
        CommandType.PIM_RED_MAX,
        CommandType.PIM_RED_MIN,
        CommandType.PIM_SCALAR_ADD,
    ]
    timings: dict[OpType, int] = {
        OpType.NOP: 1,
        OpType.SCALAR_ADD: 1,
        OpType.VEC_ADD: 1,
        OpType.VEC_SUB: 1,
        OpType.VEC_MUL: 2,
        OpType.VEC_DIV: 2,
        OpType.VEC_MAX: 1,
        OpType.VEC_MIN: 1,
        OpType.RED_ADD: 1,
        OpType.RED_MAX: 1,
        OpType.RED_MIN: 1,
        # these timings do not matter since we handle them externally
        OpType.READ: 0,
        OpType.WRITE: 0,
    }

    def __init__(
        self,
        location: tuple[int, int, int, int],
        p_mem: Ptr[MemSystem],
        registers: list[str] | None = None,
        vec_registers: list[str] | None = None,
        pipeline_stages: list[Stage] | None = None,
        tCK: float = 5.0,
    ):
        super().__init__(
            location,
            p_mem,
            registers=registers,
            vec_registers=vec_registers,
            tCK=tCK,
        )

        self.pipeline: Pipeline = Pipeline(
            self,
            (mkDefaultStages(self) if pipeline_stages is None else pipeline_stages),
        )

        self.pipeline.set_pipeline_exit_callback(self.instruction_side_effect_callback)

    @override
    def instruction_side_effect_callback(self, ins: Instruction):
        def red_form_check(ins: Instruction):
            dst = ins.in_reg1 if ins.dst == "" else ins.dst
            if len(dst) < 1 or dst not in self.registers:
                raise PimInstructionMalformedError(
                    f"Tried to map from {ins.in_reg1} data to destination: {ins.dst}. Accumulation must be sent to a register (cannot be a vector register)."
                )

        match ins.operation:
            # TODO: add appropriate form checks
            case OpType.READ | OpType.WRITE:
                self.gdl: DataWrapper = ins.ret()
                if len(ins.dst) > 0:
                    self.set_reg(ins.dst, self.gdl)
            case OpType.SCALAR_ADD:
                map_scalar_vec(self, lambda x, y: x + y, ins)
            case OpType.VEC_ADD:
                map_vec(self, lambda x, y: x + y, ins)
            case OpType.VEC_SUB:
                map_vec(self, lambda x, y: x - y, ins)
            case OpType.VEC_MUL:
                map_vec(self, lambda x, y: x * y, ins)
            case OpType.VEC_DIV:
                map_vec(self, lambda x, y: x / y, ins)
            case OpType.VEC_MAX:
                map_vec(self, max, ins)
            case OpType.RED_MAX:
                red_form_check(ins)
                dst = ins.in_reg1 if ins.dst == "" else ins.dst
                self.set_reg(dst, dtype_min(np.dtype(ins.dtype)))
                fold_vec(self, max, ins)
            case OpType.RED_ADD:
                red_form_check(ins)
                dst = ins.in_reg1 if ins.dst == "" else ins.dst
                self.set_reg(dst, np.dtype(ins.dtype)(0))
                fold_vec(self, lambda x, y: x + y, ins)
            case OpType.RED_MIN:
                red_form_check(ins)
                dst = ins.in_reg1 if ins.dst == "" else ins.dst
                self.set_reg(dst, dtype_max(np.dtype(ins.dtype)))
                fold_vec(self, min, ins)

            case _:
                pass

    def parse_cmd(self, cmd: Command) -> list[Instruction] | None:
        match cmd.cmdtype:
            case CommandType.PIM_ADD:
                vec_vec_kernel(self, cmd, OpType.VEC_ADD)
            case CommandType.PIM_SUB:
                vec_vec_kernel(self, cmd, OpType.VEC_SUB)
            case CommandType.PIM_MUL:
                vec_vec_kernel(self, cmd, OpType.VEC_MUL)
            case CommandType.PIM_DIV:
                vec_vec_kernel(self, cmd, OpType.VEC_DIV)
            case CommandType.PIM_RED_SUM:
                red_kernel(self, cmd, OpType.VEC_ADD, OpType.RED_ADD)
            case CommandType.PIM_RED_MAX:
                red_kernel(self, cmd, OpType.VEC_MAX, OpType.RED_MAX)
            case CommandType.PIM_RED_MIN:
                red_kernel(self, cmd, OpType.VEC_MIN, OpType.RED_MIN)
            case CommandType.PIM_SCALAR_ADD:
                vec_scalar_kernel(self, cmd, OpType.SCALAR_ADD)
            case _:
                raise PimCmdNotImplementedError(
                    f"PIM command type {cmd.cmdtype} not implemented for the current architeture."
                )

        return None

    @override
    def ins_queue_handler(self):
        if len(self.instruction_queue) > 0 and self.pipeline.try_load(
            self.instruction_queue[0]
        ):
            self.call_start_setter(self.instruction_queue.popleft())

    @override
    def cmd_handler(self, cmd: Command | None):
        if cmd is not None:
            if cmd.cmdtype not in self.supported_cmds:
                raise PimCmdNotSupportedError(
                    f"{self.__class__.__name__} does not support command type {cmd.cmdtype}."
                )
            # TODO: use the parsed cmd
            _ = self.parse_cmd(cmd)

    @override
    def tick(self, cmd: Command | None = None):
        self.pipeline.tick()
        super().tick(cmd)
