from lib.errors import (
    PimCmdNotSupportedError,
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
    mkEnhancedStages,
)
from lib.cores.components.functional import map_vec, fold_vec
from typing import override
import numpy as np


class Core(BaseCore):
    supported_cmds: list[CommandType] = [
        CommandType.PIM_ADD,
        CommandType.PIM_SUB,
        CommandType.PIM_DIV,
        CommandType.PIM_MUL,
        CommandType.PIM_ABS,
    ]
    isa: list[OpType] = [
        OpType.NOP,
        OpType.VEC_ADD,
        OpType.VEC_SUB,
        OpType.VEC_MUL,
        OpType.VEC_DIV,
        OpType.RED_ADD,
        OpType.READ,
        OpType.WRITE,
    ]
    timings: dict[OpType, int] = {
        OpType.NOP: 1,
        OpType.VEC_ADD: 1,
        OpType.VEC_SUB: 1,
        OpType.VEC_MUL: 2,
        OpType.VEC_DIV: 2,
        OpType.RED_ADD: 1,
        # these timings do not matter since we handle them externally
        OpType.READ: 0,
        OpType.WRITE: 0,
    }

    def __init__(
        self,
        location: tuple[int, int, int, int],
        p_mem: Ptr[MemSystem],
        scratchpad_access_time: int = 2,
        registers: list[str] | None = None,
        vec_registers: list[str] | None = None,
        pipeline_stages: list[Stage] | None = None,
        tCK: float = 1.0,
    ):
        super().__init__(
            location, p_mem, registers=registers, vec_registers=vec_registers, tCK=tCK,
        )

        self.spad_acc_time: int = scratchpad_access_time
        self.pipeline: Pipeline = Pipeline(
            self,
            (mkDefaultStages(self) if pipeline_stages is None else pipeline_stages),
        )

        self.pipeline.set_pipeline_exit_callback(self.instruction_side_effect_callback)

    @override
    def instruction_side_effect_callback(self, ins: Instruction):
        match ins.operation:
            case OpType.READ | OpType.WRITE:
                self.gdl = ins.ret()
                if len(ins.dst) > 0:
                    self.set_reg(ins.dst, self.gdl)
            case OpType.VEC_ADD:
                map_vec(self, lambda x, y: x + y, ins)
            case OpType.VEC_SUB:
                map_vec(self, lambda x, y: x - y, ins)
            case OpType.VEC_MUL:
                map_vec(self, lambda x, y: x * y, ins)
            case OpType.RED_ADD:
                dst = ins.in_reg1 if ins.dst == "" else ins.dst
                if ( len(dst) < 1 or dst not in self.registers ):
                    raise PimInstructionMalformedError(
                        f"Tried to map from {ins.in_reg1} data to destination: {ins.dst}. Accumulation must be sent to a register (cannot be a vector register)."
                    )
                else:
                    self.set_reg("regA", 0)
                    fold_vec(self, lambda x, y: x + y, ins)
            case OpType.VEC_DIV:
                map_vec(self, lambda x, y: x / y, ins)
            case _:
                pass

    def parse_cmd(self, cmd: Command) -> list[Instruction] | None:
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
