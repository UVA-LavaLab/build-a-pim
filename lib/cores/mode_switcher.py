from collections import deque
from enum import Enum
from lib.errors import (
    PimCmdNotSupportedError,
    PimCmdNotImplementedError,
    PimInstructionMalformedError,
)
from lib.memsys import MemSystem
from lib.cores.instructions import Instruction, OpType
from lib.cores.components.base import BaseCore
from lib.monad import Ptr
from lib.controller.commands import CommandType, Command
from lib.cores.components.pipeline import (
    Stage,
    Pipeline,
    mkDefaultStages,
)
from typing import override
from lib.controller.response import Response
from lib.monad import DataWrapper
import numpy as np
import numpy.typing as npt
import math


class Mode(Enum):
    PIM = 0
    DIRTY = 1
    PAUSED = 2


class Core(BaseCore):
    supported_cmds: list[CommandType] = [
        CommandType.SWITCH_MODE_MEM,
        CommandType.SWITCH_MODE_PIM,
        CommandType.PIM_BANK_PING,
        CommandType.MEM_READ,
    ]
    timings: dict[OpType, int] = {
        OpType.NOP: 1,
        OpType.TO_SWITCHING_MODE: 1,
        OpType.TO_PIM_MODE: 1,
        OpType.TO_PAUSED_MODE: 1,
        # these timings do not matter since we handle them externally
        OpType.GET_ACTIVE_ROW: 0,
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
        self.responses: deque[Response] = deque()
        self.paused: bool = False
        # set the default mode to PIM mode
        self.mode: Mode = Mode.PIM
        self.timings[OpType.GET_ACTIVE_ROW] = self.p_mem().get_config_param("BIL")
        self.emit_count: int = 0

    def emit(self, data: DataWrapper):
        self.responses.append(
            Response(
                p_mem=self.p_mem,
                response_bits=self.p_mem().m_gdl_width,
                data=np.frombuffer(data.data, dtype=np.uint8),
            )
        )
        self.emit_count -= 1

    @override
    def instruction_side_effect_callback(self, ins: Instruction):
        match ins.operation:
            case OpType.READ | OpType.WRITE:
                if self.emit_count > 0 and ins.emit:
                    if self.mode == Mode.PIM:
                        raise PimInstructionMalformedError(
                            "Cannot emit from an instruction in PIM mode."
                        )
                    self.emit(ins.ret())
                self.gdl = ins.ret()
                if len(ins.dst) > 0:
                    self.set_reg(ins.dst, self.gdl)
            case OpType.GET_ACTIVE_ROW:
                self.responses.append(
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
                )
            case _:
                pass

    def add_emitted_instruction(
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
                emit=True,
            )
        )

    def parse_cmd(self, cmd: Command) -> list[Instruction] | Response | None:
        match cmd.cmdtype:
            case CommandType.MEM_READ:
                if (
                    cmd.location[0] == self.channel
                    and cmd.location[1] == self.rank
                    and cmd.location[2] == self.bankgroup
                ):
                    self.emit_count += 1
                    # NOTE: in a fully-fledged version, this would need to be
                    # implemented as a prepend operation to the instruction queue
                    if self.bank == cmd.location[3]:
                        self.add_emitted_instruction(OpType.READ, addr=cmd.addr)
                    else:
                        self.add_instruction(OpType.READ, addr=cmd.addr)
                else:
                    # add a nop to take the place of the hypothetical compare
                    # instruction
                    self.add_instruction(OpType.NOP)
            case CommandType.PIM_BANK_PING:
                # only return a response when the
                # bank information is not relevant
                # to this mode switching implementation

                if (
                    self.bankgroup == cmd.location[2]
                    and self.rank == cmd.location[1]
                    and self.channel == cmd.location[0]
                ):
                    self.add_instruction(OpType.GET_ACTIVE_ROW)
            case CommandType.SWITCH_MODE_MEM:
                # problem: when changing (to mem), we need to ensure that
                # the pipeline is empty before executing a memory operation
                # or changing the mode on the mem object
                self.add_instruction(OpType.TO_SWITCHING_MODE)
            case CommandType.SWITCH_MODE_PIM:
                # problem: when changing modes (to pim), we need to ensure that
                # all currently executing transactions are done being handled *and*
                # stop new ones from populating the queue
                self.add_instruction(OpType.TO_PIM_MODE)
            case _:
                raise PimCmdNotImplementedError(
                    f"PIM command type {cmd.cmdtype} not implemented for the current architeture."
                )

        return None

    # FIXME: fix this, as it no longer reflects a real use case
    def mem_mode_ready(self) -> bool:
        return self.paused and self.pipeline.is_empty()

    def switch_mode(self, op: OpType):
        match op:
            case OpType.TO_PAUSED_MODE:
                self.mode = Mode.PAUSED
            case OpType.TO_PIM_MODE:
                self.mode = Mode.PIM
            case OpType.TO_SWITCHING_MODE:
                self.mode = Mode.DIRTY
            case _:
                return

    @override
    def ins_queue_handler(self):
        if len(self.instruction_queue) > 0:
            if self.mode == Mode.PIM and self.pipeline.try_load(
                self.instruction_queue[0]
            ):
                ins = self.instruction_queue.popleft()
                # change modes when appropriate
                if ins.operation in [OpType.TO_PAUSED_MODE, OpType.TO_SWITCHING_MODE]:
                    self.switch_mode(ins.operation)
                self.call_start_setter(ins)
            elif self.mode == Mode.DIRTY and self.instruction_queue[0].operation in [
                OpType.GET_ACTIVE_ROW,
                OpType.TO_SWITCHING_MODE,
                OpType.TO_PAUSED_MODE,
                OpType.READ,
                OpType.NOP,
            ]:
                # if we can load the instruction into the pipeline, then we
                # register its side effects
                if self.pipeline.try_load(self.instruction_queue[0]):
                    ins = self.instruction_queue.popleft()
                    if ins.operation != OpType.GET_ACTIVE_ROW:
                        self.switch_mode(ins.operation)
                    self.call_start_setter(ins)
            elif (
                self.mode == Mode.DIRTY
                and self.instruction_queue[0].operation == OpType.TO_PIM_MODE
                and self.pipeline.is_empty()
            ):
                # separate handling for switching back to pim mode (makes sure
                # that all mem transactions finish)
                _ = self.pipeline.try_load(self.instruction_queue[0])
                ins = self.instruction_queue.popleft()
                self.switch_mode(ins.operation)

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
        # return one response per cycle
        if len(self.responses) > 0:
            return self.responses.popleft()
