from collections import deque
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
    map_vec,
    fold_vec,
    red_kernel,
)
from typing import override, Callable
from lib.controller.response import Response
import numpy as np
import math


class Core(BaseCore):
    supported_cmds: list[CommandType] = [
        CommandType.SWITCH_MODE_MEM,
        CommandType.SWITCH_MODE_PIM,
        CommandType.PIM_BANK_PING,
    ]
    timings: dict[OpType, int] = {
        OpType.NOP: 1,
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
        tCK: float = 5.0,
    ):
        super().__init__(
            location,
            p_mem,
            registers=registers,
            vec_registers=vec_registers,
            tCK=tCK,
        )

        self.spad_acc_time: int = scratchpad_access_time
        self.pipeline: Pipeline = Pipeline(
            self,
            (mkDefaultStages(self) if pipeline_stages is None else pipeline_stages),
        )

        self.pipeline.set_pipeline_exit_callback(self.instruction_side_effect_callback)
        self.responses: deque[Response] = deque()
        self.paused: bool = False

    @override
    def instruction_side_effect_callback(self, ins: Instruction):
        def red_form_check(ins: Instruction):
            dst = ins.in_reg1 if ins.dst == "" else ins.dst
            if len(dst) < 1 or dst not in self.registers:
                raise PimInstructionMalformedError(
                    f"Tried to map from {ins.in_reg1} data to destination: {ins.dst}. Accumulation must be sent to a register (cannot be a vector register)."
                )

        match ins.operation:
            case OpType.READ | OpType.WRITE:
                self.gdl = ins.ret()
                if len(ins.dst) > 0:
                    self.set_reg(ins.dst, self.gdl)
            case _:
                pass

    def parse_cmd(self, cmd: Command) -> list[Instruction] | Response | None:
        match cmd.cmdtype:
            case CommandType.PIM_BANK_PING:
                # only return a response when the 
                return (
                    Response(
                        self.p_mem,
                        # count the number of bits required to represent the currently active row.
                        # represents the minimum number of encodings needed to represent each
                        # state in the bank
                        response_bits=self.p_mem()
                        .get_config_param("ro_mask")
                        .bit_count(),
                        active_row=self.p_mem().get_active_row(
                            self.channel, self.rank, self.bankgroup, self.bank
                        ),
                        # this data can be communicated implicitly, thus does not contribute
                        # to the overall size of the response packet
                        bank=self.bank,
                    )
                    # bank information is not relevant 
                    # to this mode switching implementation
                    if self.bankgroup == cmd.location[2]
                    and self.rank == cmd.location[1]
                    and self.channel == cmd.location[0]
                    else None
                )
            case CommandType.SWITCH_MODE_MEM:
                # problem: when changing (to mem), we need to ensure that
                # the pipeline is empty before executing a memory operation 
                # or changing the mode on the mem object
                self.paused = True
            case CommandType.SWITCH_MODE_PIM:
                # problem: when changing modes (to pim), we need to ensure that
                # all currently executing transactions are done being handled *and*
                # stop new ones from populating the queue
                self.paused = False
            case _:
                raise PimCmdNotImplementedError(
                    f"PIM command type {cmd.cmdtype} not implemented for the current architeture."
                )

        return None

    def mem_mode_ready(self) -> bool:
        return self.paused and self.pipeline.is_empty()

    @override
    def ins_queue_handler(self):
        if not self.paused and len(self.instruction_queue) > 0 and self.pipeline.try_load(
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
            response = self.parse_cmd(cmd)
            if isinstance(response, Response):
                self.responses.append(response)

    @override
    def tick(self, cmd: Command | None = None) -> Response | None:
        self.pipeline.tick()
        _ = super().tick(cmd)
        if len(self.responses) > 0:
            return self.responses.popleft()
