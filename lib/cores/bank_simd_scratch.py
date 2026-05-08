from lib.errors import (
    PimCmdNotSupportedError,
    PimCmdNotImplementedError,
    PimInstructionMalformedError,
)
from lib.memsys import MemSystem
from lib.cores.instructions import IState, Instruction, OpType
from lib.cores.components.base import BaseCore
from lib.cores.components.scratchpad import Scratchpad
from lib.containers import Box, Ptr
from lib.controller.commands import CommandType, Command
from lib.cores.components.pipeline import (
    Stage,
    Pipeline,
    mkDefaultStages,
)
from lib.cores.components.functional import (
    conditional_jump,
    dtype_min,
    dtype_max,
    imm_operation,
    map_scalar_vec,
    map_vec,
    fold_vec,
    red_kernel,
    vec_scalar_kernel,
    vec_vec_kernel,
)
from lib.cores.components.instruction_cache import InstructionCache
from typing import override, Callable
import numpy as np
import math


class Core(BaseCore):
    """
    A bank-level SIMD core with an associated scratchpad and modeled
    instruction cache.
    """
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
        OpType.JMP: 1,
        OpType.JL: 1,
        OpType.JGE: 1,
        OpType.MOV: 1,
        OpType.NOP: 1,
        OpType.SCALAR_ADD: 1,
        OpType.IMM_ADD: 1,
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
        OpType.SCRATCH_READ: 0,
        OpType.SCRATCH_WRITE: 0,
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

        self.scratchpad: Scratchpad = Scratchpad(bus_width=self.p_mem().m_gdl_width)
        self.pipeline: Pipeline = Pipeline(
            self,
            (mkDefaultStages(self) if pipeline_stages is None else pipeline_stages),
        )

        self.pipeline.set_pipeline_exit_callback(self.instruction_side_effect_callback)
        self.instruction_cache: InstructionCache = InstructionCache()
        self.pc: int = 0

    @override
    def instruction_side_effect_callback(self, ins: Instruction):
        def red_form_check(ins: Instruction):
            dst = ins.in_reg1 if ins.dst == "" else ins.dst
            if len(dst) < 1 or dst not in self.registers:
                raise PimInstructionMalformedError(
                    f"Tried to map from {ins.in_reg1} data to destination: {ins.dst}."
                    + f"Accumulation must be sent to a register (cannot be a vector register)."
                )

        match ins.operation:
            # TODO: add appropriate form checks
            case OpType.READ | OpType.WRITE:
                self.gdl: Box = ins.ret()
                if len(ins.dst) > 0:
                    self.set_reg(ins.dst, self.gdl)
            case OpType.MOV:
                if ins.in_reg1 == "":
                    self.set_reg(ins.dst, ins.imm)
                else:
                    self.set_reg(ins.dst, ins.get_state_by_operand_id(0))
            case OpType.IMM_ADD:
                imm_operation(self, lambda x, y: x + y, ins)
            case OpType.SCRATCH_READ:
                self.set_reg(ins.dst, ins.ret())
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
                self.set_reg(dst, np.dtype(ins.dtype).type(0))
                fold_vec(self, lambda x, y: x + y, ins)
            case OpType.RED_MIN:
                red_form_check(ins)
                dst = ins.in_reg1 if ins.dst == "" else ins.dst
                self.set_reg(dst, dtype_max(np.dtype(ins.dtype)))
                fold_vec(self, min, ins)
            case _:
                return

    def parse_cmd(self, cmd: Command) -> list[Instruction] | None:
        match cmd.cmdtype:
            case CommandType.PIM_ADD:
                return vec_vec_kernel(self, cmd, OpType.VEC_ADD)
            case CommandType.PIM_SUB:
                return vec_vec_kernel(self, cmd, OpType.VEC_SUB)
            case CommandType.PIM_MUL:
                return vec_vec_kernel(self, cmd, OpType.VEC_MUL)
            case CommandType.PIM_DIV:
                return vec_vec_kernel(self, cmd, OpType.VEC_DIV)
            case CommandType.PIM_RED_SUM:
                return red_kernel(self, cmd, OpType.VEC_ADD, OpType.RED_ADD)
            case CommandType.PIM_RED_MAX:
                return red_kernel(self, cmd, OpType.VEC_MAX, OpType.RED_MAX)
            case CommandType.PIM_RED_MIN:
                return red_kernel(self, cmd, OpType.VEC_MIN, OpType.RED_MIN)
            case CommandType.PIM_SCALAR_ADD:
                return vec_scalar_kernel(self, cmd, OpType.SCALAR_ADD)
            case _:
                raise PimCmdNotImplementedError(
                    f"PIM command type {cmd.cmdtype} not implemented for the current architecture."
                )

        return None

    @override
    def ins_queue_handler(self):
        ins: Instruction | None = self.instruction_cache[self.pc]
        if ins is not None and self.pipeline.try_load(ins):
            self.pc += 1
            # wrap PC around address space in the event of overflow
            if self.pc >= self.instruction_cache.size:
                self.pc = 0

            # in the event of an unconditional jump, we can instantly fetch the
            # next instruction. this may be an unrealistic assumption, but we
            # will revise this once we have data to confirm this decision
            if ins.operation == OpType.JMP:
                assert ins.addr != -1
                self.pc = ins.addr

            def ifail(cond: bool, errmsg: str):
                if cond:
                    raise Exception(errmsg)

            match ins.operation:
                case OpType.SCRATCH_READ:
                    ifail(
                        ins.addr <= -1,
                        f"No address supplied for instruction {ins.operation}.",
                    )
                    ifail(
                        ins.in_reg1 != "" or ins.in_reg2 != "",
                        f"Undefined behavior: one or more input registers are set for {ins.operation} instruction.",
                    )

                    def scb():
                        ins.data = self.scratchpad.read_bytes(self, ins.addr)

                    ins.start_cb = scb
                    ins.set_is_done(lambda: ins.data.is_ready())
                case OpType.SCRATCH_WRITE:
                    ifail(
                        ins.addr <= -1,
                        f"No address supplied for instruction {ins.operation}.",
                    )
                    ifail(
                        ins.in_reg1 == "", f"No register supplied to {ins.operation}."
                    )

                    def scb():
                        ins.data = self.scratchpad.write_bytes(
                            self, ins.addr, self.get_reg(ins.in_reg1)
                        )

                    ins.start_cb = scb
                    ins.set_is_done(lambda: ins.data.is_ready())
                case _:
                    self.call_start_setter(ins)

    @override
    def cmd_handler(self, cmd: Command | None):
        if cmd is not None:
            if cmd.cmdtype not in self.supported_cmds:
                raise PimCmdNotSupportedError(
                    f"{self.__class__.__name__} does not support command type {cmd.cmdtype}."
                )
            prog = self.parse_cmd(cmd)
            if prog is not None:
                # FIXME: can crash program if overflow occurs
                _ = self.load_program(prog)

    @override
    def call_start_setter(self, ins: Instruction):
        def new_is_done(ins: Instruction, update: Callable[[], bool]):
            condition: bool = ins.completion_time <= 0
            if condition and ins.state != IState.DONE:
                if update():
                    for s in self.pipeline.stages:
                        s.flush()
                        if s.name == "execute":
                            break
            return condition

        # we need ensure that jump effects occur *exactly* when the instruction
        # is done to avoid unnecesary overheads, so we can't affort to wait until
        # instruction commit time
        match ins.operation:
            case OpType.JG:
                ins.set_is_done(
                    lambda: new_is_done(
                        ins,
                        lambda: conditional_jump(
                            self, lambda x, y: x > y, ins, ins.dtype
                        ),
                    )
                )
            case OpType.JGE:
                ins.set_is_done(
                    lambda: new_is_done(
                        ins,
                        lambda: conditional_jump(
                            self, lambda x, y: x >= y, ins, ins.dtype
                        ),
                    )
                )
            case OpType.JL:
                ins.set_is_done(
                    lambda: new_is_done(
                        ins,
                        lambda: conditional_jump(
                            self, lambda x, y: x < y, ins, ins.dtype
                        ),
                    )
                )
            case OpType.JLE:
                ins.set_is_done(
                    lambda: new_is_done(
                        ins,
                        lambda: conditional_jump(
                            self, lambda x, y: x <= y, ins, ins.dtype
                        ),
                    )
                )
            case OpType.JNE:
                ins.set_is_done(
                    lambda: new_is_done(
                        ins,
                        lambda: conditional_jump(
                            self, lambda x, y: x != y, ins, ins.dtype
                        ),
                    )
                )
            case _:
                pass
        super().call_start_setter(ins)

    def load_program(self, prog: list[Instruction]) -> tuple[int, bool]:
        return self.instruction_cache.load_prog(prog)

    @override
    def tick(self, cmd: Command | None = None):
        self.pipeline.tick()
        _ = super().tick(cmd)
