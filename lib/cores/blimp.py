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
    ]
    timings: dict[OpType, int] = {
        OpType.NOP: 1,
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
            case OpType.VEC_ADD:
                # TODO: add a form check for vector instructions
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
                # General algorithm:
                # For each vector register (cap at the size of a row in GDL-width chunks),
                #       load a chunk of the first input into the register file.
                # Then switch to input 2, load one chunk per vector register and add to the same register
                # Then switch to destination range and write all of the chunks
                # Repeat until the end of the vector
                # FIXME: this does NOT account for PIM objects which wrap around the address space...
                # FIXME: there are also some explorations to be done regarding whether it is faster
                # to have a HUGE register file and load both vectors into that, then accumulate between them from there

                # safety checks
                assert (
                    cmd.range_1[1] - cmd.range_1[0] == cmd.range_2[1] - cmd.range_2[0]
                )
                # by the transitive property of equality, we don't need to check the last pair
                # TODO: figure out how to programmatically relax this to allow for any ratio of input to output sizes
                # this will faciliate compression and binary operations
                assert (
                    cmd.range_1[1] - cmd.range_1[0]
                    == cmd.range_dst[1] - cmd.range_dst[0]
                )

                # window size is the number of chunks we can calculate
                # without overflowing the available vector registers
                window_size_chunks = min(
                    len(self.vec_registers), self.p_mem().get_config_param("n_col")
                )

                gdl_size_bytes = self.p_mem().m_gdl_width / 8
                to_chunk_range: Callable[[tuple[int, int]], tuple[int, int]] = (
                    lambda t: (int(t[0] / gdl_size_bytes), int(t[1] / gdl_size_bytes))
                )
                i1_range = to_chunk_range(cmd.range_1)
                i2_range = to_chunk_range(cmd.range_2)
                dst_range = to_chunk_range(cmd.range_dst)
                input_chunks = i1_range[1] - i1_range[0]

                for w in range(math.ceil(input_chunks / window_size_chunks)):
                    # read the batch from the first vector
                    for c, b in enumerate(range(
                        i1_range[0] + w * window_size_chunks,
                        i1_range[0] + (w + 1) * window_size_chunks,
                    )):
                        target_reg = self.vec_registers[c]
                        self.add_instruction(
                            OpType.READ, addr=b, dst=target_reg, dtype=cmd.dtype
                        )
                    # read the batch from the second vector and accumulate to the first register locations
                    for c, b in enumerate(range(
                        i2_range[0] + w * window_size_chunks,
                        i2_range[0] + (w + 1) * window_size_chunks,
                    )):
                        self.add_instruction(OpType.READ, addr=b, dtype=cmd.dtype)
                        target_reg = self.vec_registers[c]
                        self.add_instruction(
                            OpType.VEC_ADD,
                            in_reg1=target_reg,
                            in_reg2="gdl",
                            dtype=cmd.dtype,
                        )
                    # write back to core-local memory at the appropriate index
                    for b in range(
                        dst_range[0] + w * window_size_chunks,
                        dst_range[0] + (w + 1) * window_size_chunks,
                    ):
                        self.add_instruction(OpType.WRITE, addr=b, dtype=cmd.dtype)
                # print("\n".join([str(i) for i in self.instruction_queue]))
                # raise PimCmdNotImplementedError("PIM_ADD not done with implementation")
            case CommandType.PIM_RED_SUM:
                red_kernel(self, cmd, OpType.VEC_ADD, OpType.RED_ADD)
            case CommandType.PIM_RED_MAX:
                red_kernel(self, cmd, OpType.VEC_MAX, OpType.RED_MAX)
            case CommandType.PIM_RED_MIN:
                red_kernel(self, cmd, OpType.VEC_MIN, OpType.RED_MIN)
                # print("\n".join([str(ins) for ins in self.instruction_queue]))
                # raise Exception()
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
