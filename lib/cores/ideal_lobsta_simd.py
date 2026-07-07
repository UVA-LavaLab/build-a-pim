from math import log2
from lib.cores.components.addressing import Swizzler
from lib.errors import (
    PimCmdNotSupportedError,
    PimCmdNotImplementedError,
    PimInstructionMalformedError,
    HardwareSupportError,
    AutoqueueAppendError,
)
from lib.memsys import MemSystem
from lib.cores.instructions import IState, Instruction, OpType
from lib.cores.components.base import BaseCore
from lib.cores.components.scratchpad import Scratchpad
from lib.cores.components.buffer import DataBuffer
from lib.containers import Box, Ptr
from lib.util import rev_enum
from lib.controller.commands import CommandType, Command
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
    streamed_red_kernel,
    streamed_vec_scalar_kernel,
    streamed_vec_vec_kernel,
)
from lib.cores.components.instruction_cache import InstructionCache
from typing import override, Callable
from collections import deque
import numpy as np


# TODO: determine overhead of routing this through the bank interface vs putting it at the quartarray level
class Autoqueue:
    def __init__(self, core: Core, quartarray: int = 0):
        self.scratchpad: Scratchpad = Scratchpad(
            2048, bus_width=core.p_mem().m_gdl_width
        )
        self.size: int = int(self.scratchpad.size / (core.p_mem().m_gdl_width / 8))
        self.head: int = 0
        self.tail: int = 0
        self.core: Core = core
        self.contracts: deque[Box] = deque()
        self.sp_contracts: deque[Box] = deque()
        self.target: int = 0
        self.base: int = 0
        self.quartarray: int = max(min(quartarray, self.core.sub_rows - 1), 0)

    def succ(self, i: int) -> int:
        return (i + 1) % self.size

    def inc_head(self) -> bool:
        new_head = self.succ(self.head)
        if new_head != self.tail:
            self.head = new_head
            return True
        return False

    def inc_tail(self) -> bool:
        new_tail = self.succ(self.tail)
        if self.tail != self.head:
            self.tail = new_tail
            return True
        return False

    def pop(self) -> Box | None:
        old_tail: int = self.tail
        if self.inc_tail():
            return self.scratchpad.read_bytes(self.core, old_tail)
        else:
            return None

    def push(self, boxed_data: Box) -> bool:
        old_head = self.head
        if self.inc_head():
            self.sp_contracts.append(
                self.scratchpad.write_bytes(
                    self.core, old_head, np.frombuffer(boxed_data.data, dtype=np.uint8)
                )
            )
            return True
        return False

    def set_base(self, addr: int):
        """
        Set the starting point from which the autoqueue should begin loading
        data from its local quartarray.

        Translates a global LoBSTA address to the global address space, then
        sets the queue's base to the translated local address.
        """
        addr = self.core.buf.swizzler(addr)
        _, _, _, _, la = self.core.p_mem().loc_from_addr(addr)
        self.base = la

    def set_target(self, addr: int):
        """
        Set the endpoint at which the autoqueue should stop loading data from
        its local quartarray.

        Translates a global LoBSTA address to the global address space, then
        sets the queue's target to the translated local address.
        """
        addr = self.core.buf.swizzler(addr)
        _, _, _, _, la = self.core.p_mem().loc_from_addr(addr)
        self.target = la

    def is_full(self):
        reserved_head = (self.head + len(self.contracts)) % self.size
        return self.succ(reserved_head) == self.tail

    def tick(self):
        # mem logic
        if self.base < self.target:
            # if we pass the address check, check fullness
            if not self.is_full():
                # now, we know we can fetch the data
                ch, ra, bg, ba = self.core.location
                self.contracts.append(
                    self.core.buf.get(
                        (ch, ra, bg, ba * self.core.sub_rows + self.quartarray),
                        self.base,
                        swizzle=False,
                    )
                )
                self.base += 1

        # scratchpad logic, must be fifo (queue)
        if len(self.contracts) > 0 and self.contracts[0].is_ready():
            cont = self.contracts.popleft()
            if not self.push(cont):
                raise AutoqueueAppendError(
                    f"Failed to append to autoqueue in core placed at"
                    + f"(ch, ra, bg, ba): {self.core.location}."
                )

        while len(self.sp_contracts) > 0 and self.sp_contracts[0].is_ready():
            _ = self.sp_contracts.popleft()


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
        # TODO: actually support these
        CommandType.PIM_SCALAR_SUB,
        CommandType.PIM_SCALAR_MUL,
        CommandType.PIM_SCALAR_DIV,
    ]
    timings: dict[OpType, int] = {
        OpType.JMP: 1,
        OpType.JL: 1,
        OpType.JGE: 1,
        OpType.MOV: 1,
        OpType.NOP: 1,
        OpType.SCALAR_ADD: 1,
        OpType.SCALAR_SUB: 1,
        OpType.SCALAR_MUL: 1,
        OpType.SCALAR_DIV: 1,
        OpType.IMM_ADD: 1,
        OpType.IMM_SUB: 1,
        OpType.IMM_MUL: 1,
        OpType.IMM_DIV: 1,
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
        OpType.EAGER_ACTIVATE: 0,
    }

    def __init__(
        self,
        location: tuple[int, int, int, int],
        p_mem: Ptr[MemSystem],
        registers: list[str] | None = None,
        vec_registers: list[str] | None = None,
        num_registers: int = 26,
        num_vec_registers: int = 26,
        streaming: bool = False,
        sub_rows: int = 4,
        tCK: float = 5.0,
    ):
        super().__init__(
            location,
            p_mem,
            registers=registers,
            vec_registers=vec_registers,
            num_registers=num_registers,
            num_vec_registers=num_vec_registers,
            tCK=tCK,
        )
        # core configuration variables
        self.streaming: bool = streaming
        self.sub_rows: int = sub_rows
        if sub_rows & (sub_rows - 1) != 0:
            raise HardwareSupportError("Number of sub-rows must be a power of 2.")

        # temp variables to help define later operations
        ro: int = p_mem().get_config_param("ro_pos") + p_mem().get_config_param(
            "shift_bits"
        )
        ba: int = p_mem().get_config_param("ba_pos") + p_mem().get_config_param(
            "shift_bits"
        )
        shift: int = int(log2(sub_rows))

        row_mask = p_mem().get_config_param("ro_mask") << ro
        bank_selection_mask = (sub_rows - 1) << ba

        # address swizzling function
        # ensures first-col, then quartarray, then row iteration for sequential
        # addresses within the core's local address space
        def swizzle(addr: int) -> int:
            row = (addr & row_mask) >> ro
            sub_row = row & (sub_rows - 1)
            physical_row = row >> shift

            addr &= ~row_mask
            addr &= ~bank_selection_mask

            addr |= physical_row << ro
            addr |= sub_row << ba
            return addr

        # creates a data buffer which automatically swizzles addresses.
        self.buf: DataBuffer = DataBuffer(
            self.p_mem,
            swizzler=Swizzler(swap_functions=[swizzle]),
        )

        # instruction control definitions
        self.instruction_cache: InstructionCache = InstructionCache()
        self.ins_stream: list[Instruction] = []
        self.pc: int = 0
        self.ins_queue: deque[Instruction] = deque()
        self.active_ins: list[Instruction] = []

        # scratchpad declarations
        self.ptr_q: Autoqueue = Autoqueue(self)
        self.idx_q: Autoqueue = Autoqueue(self)
        self.scratchpad: Scratchpad = Scratchpad(bus_width=self.p_mem().m_gdl_width)

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
                # self.gdl: Box = ins.ret()
                if len(ins.dst) > 0:
                    self.set_reg(ins.dst, ins.ret())
                else:
                    self.gdl: Box = ins.ret()
            case OpType.MOV:
                if ins.in_reg1 == "":
                    self.set_reg(ins.dst, ins.imm)
                else:
                    self.set_reg(ins.dst, ins.get_state_by_operand_id(0))
            case OpType.IMM_ADD:
                imm_operation(self, lambda x, y: x + y, ins)
            case OpType.IMM_SUB:
                imm_operation(self, lambda x, y: x - y, ins)
            case OpType.IMM_MUL:
                imm_operation(self, lambda x, y: x * y, ins)
            case OpType.IMM_DIV:
                imm_operation(self, lambda x, y: x / y, ins)
            case OpType.SCRATCH_READ:
                self.set_reg(ins.dst, ins.ret())
            case OpType.SCALAR_ADD:
                map_scalar_vec(self, lambda x, y: x + y, ins)
            case OpType.SCALAR_SUB:
                map_scalar_vec(self, lambda x, y: x - y, ins)
            case OpType.SCALAR_MUL:
                map_scalar_vec(self, lambda x, y: x * y, ins)
            case OpType.SCALAR_DIV:
                map_scalar_vec(self, lambda x, y: x / y, ins)
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
                return (
                    vec_vec_kernel(self, cmd, OpType.VEC_ADD)
                    if not self.streaming
                    else streamed_vec_vec_kernel(self, cmd, OpType.VEC_ADD)
                )
            case CommandType.PIM_SUB:
                return (
                    vec_vec_kernel(self, cmd, OpType.VEC_SUB)
                    if not self.streaming
                    else streamed_vec_vec_kernel(self, cmd, OpType.VEC_SUB)
                )
            case CommandType.PIM_MUL:
                return (
                    vec_vec_kernel(self, cmd, OpType.VEC_MUL)
                    if not self.streaming
                    else streamed_vec_vec_kernel(self, cmd, OpType.VEC_MUL)
                )
            case CommandType.PIM_DIV:
                return (
                    vec_vec_kernel(self, cmd, OpType.VEC_DIV)
                    if not self.streaming
                    else streamed_vec_vec_kernel(self, cmd, OpType.VEC_DIV)
                )
            case CommandType.PIM_RED_SUM:
                return (
                    red_kernel(self, cmd, OpType.VEC_ADD, OpType.RED_ADD)
                    if not self.streaming
                    else streamed_red_kernel(self, cmd, OpType.VEC_ADD, OpType.RED_ADD)
                )
            case CommandType.PIM_RED_MAX:
                return (
                    red_kernel(self, cmd, OpType.VEC_MAX, OpType.RED_MAX)
                    if not self.streaming
                    else streamed_red_kernel(self, cmd, OpType.VEC_MAX, OpType.RED_MAX)
                )
            case CommandType.PIM_RED_MIN:
                return (
                    red_kernel(self, cmd, OpType.VEC_MIN, OpType.RED_MIN)
                    if not self.streaming
                    else streamed_red_kernel(self, cmd, OpType.VEC_MIN, OpType.RED_MIN)
                )
            case CommandType.PIM_SCALAR_ADD:
                return (
                    vec_scalar_kernel(self, cmd, OpType.SCALAR_ADD)
                    if not self.streaming
                    else streamed_vec_scalar_kernel(self, cmd, OpType.SCALAR_ADD)
                )
            case CommandType.PIM_SCALAR_SUB:
                return (
                    vec_scalar_kernel(self, cmd, OpType.SCALAR_SUB)
                    if not self.streaming
                    else streamed_vec_scalar_kernel(self, cmd, OpType.SCALAR_SUB)
                )
            case CommandType.PIM_SCALAR_MUL:
                return (
                    vec_scalar_kernel(self, cmd, OpType.SCALAR_MUL)
                    if not self.streaming
                    else streamed_vec_scalar_kernel(self, cmd, OpType.SCALAR_MUL)
                )
            case CommandType.PIM_SCALAR_DIV:
                return (
                    vec_scalar_kernel(self, cmd, OpType.SCALAR_DIV)
                    if not self.streaming
                    else streamed_vec_scalar_kernel(self, cmd, OpType.SCALAR_DIV)
                )
            case _:
                raise PimCmdNotImplementedError(
                    f"PIM command type {cmd.cmdtype} not implemented for the current architecture."
                )

        return None

    @override
    def ins_queue_handler(self):
        """
        Handles instruction queue logic. Instructions are read out of the
        instruction cache and loaded into the instruction queue (mimics a fetch
        phase).
        """
        ins: Instruction | None = (
            self.instruction_cache[self.pc]
            if not self.streaming
            else (self.ins_stream[self.pc] if self.pc < len(self.ins_stream) else None)
        )
        if ins is not None and not any([ins.is_jump() for ins in self.active_ins]):
            self.ins_queue.append(ins)
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
            # condition: bool = ins.completion_time <= 0
            # if condition and ins.state != IState.DONE:
            #
            # self.active_ins = []
            _ = update()
            return True

        def ifail(cond: bool, errmsg: str):
            if cond:
                raise Exception(errmsg)

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
            case OpType.EAGER_ACTIVATE:
                ifail(
                    ins.addr <= -1,
                    "No address supplied for instruction READ.",
                )
                ifail(
                    ins.in_reg2 != "",
                    "Undefined behavior: secondary input register (in_reg2) set for READ instruction.",
                )

                def scb():
                    offset = (
                        int(ins.get_state_by_operand_id(0)) if ins.in_reg1 != "" else 0
                    )
                    ins.data = self.buf.get(
                        (
                            self.channel,
                            self.rank,
                            self.bankgroup,
                            self.bank * self.sub_rows,
                            # makes the interpreter not freak out
                        ),
                        int(ins.addr) + offset,
                    )

                ins.start_cb = scb
                ins.set_is_done(lambda: ins.data.is_ready())

            case OpType.READ:
                ifail(
                    ins.addr <= -1,
                    "No address supplied for instruction READ.",
                )
                ifail(
                    ins.in_reg2 != "",
                    "Undefined behavior: secondary input register (in_reg2) set for READ instruction.",
                )

                def scb():
                    offset = (
                        int(ins.get_state_by_operand_id(0)) if ins.in_reg1 != "" else 0
                    )
                    ins.data = self.buf.get(
                        (
                            self.channel,
                            self.rank,
                            self.bankgroup,
                            self.bank * self.sub_rows,
                            # makes the interpreter not freak out
                        ),
                        int(ins.addr) + offset,
                    )

                ins.start_cb = scb
                ins.set_is_done(lambda: ins.data.is_ready())

            case OpType.WRITE:
                ifail(
                    ins.addr <= -1,
                    "No address supplied for instruction WRITE.",
                )
                ifail(
                    ins.in_reg2 != "" and ins.in_reg2 not in self.registers,
                    "Undefined behavior: Secondary input register (in_reg2) is not a scalar"
                    + "register; behavior not defined.",
                )

                def scb():
                    offset = (
                        int(ins.get_state_by_operand_id(1)) if ins.in_reg2 != "" else 0
                    )
                    dst: Box = (
                        ins.get_state_by_operand_id(0)
                        if ins.in_reg1 != ""
                        else self.gdl
                    )

                    ins.data = self.buf.set(
                        (
                            self.channel,
                            self.rank,
                            self.bankgroup,
                            self.bank * self.sub_rows,
                        ),
                        int(ins.addr) + offset,
                        dst,
                    )

                ins.start_cb = scb
                ins.set_is_done(lambda: ins.data.is_ready())
            case _:
                pass

    def load_program(self, prog: list[Instruction]) -> tuple[int, bool]:
        if self.streaming:
            self.ins_stream = self.ins_stream + prog
            return (-1, True)
        return self.instruction_cache.load_prog(prog)

    def has_hazard(self, ins: Instruction):
        """
        Detects if a passed instruction has a hazard in the active instruction
        list.
        """
        inputs = ins.in_set()
        for active in self.active_ins:
            if inputs & active.out_set():
                return True
        return False

    def active_ins_handler(self):
        """
        This function handles the active instruction logic. It makes sure that
        data is supplied to instructions in a program-order fashion and removes
        active instructions that have terminated, progressing those which can
        be progressed.
        """
        if len(self.ins_queue) > 0 and (
            self.ins_queue[0].is_mem() or len(self.active_ins) == 0
        ):
            can_issue = self.ins_queue[0].is_mem() or len(self.active_ins) == 0

            if can_issue and not self.has_hazard(self.ins_queue[0]):
                ins = self.ins_queue.popleft()
                # capture operands
                for op in [ins.in_reg1, ins.in_reg2]:
                    # for the NOP case
                    if op != "":
                        ins.set_state_by_operand_name(op, self.get_reg(op))
                self.active_ins.append(ins)

        for i, ins in rev_enum(self.active_ins):
            if ins.is_cold():
                ins.start()
            else:
                ins.tick()

            if ins.is_done():
                self.instruction_side_effect_callback(self.active_ins[i])
                del self.active_ins[i]

    @override
    def is_idle(self) -> bool:
        return len(self.active_ins) == 0 and len(self.ins_queue) == 0

    @override
    def tick(self, cmd: Command | None = None):
        # On each tick, we want to: issue any and all memory requests (infinite
        # buffer size, since we can assume a real device would buffer
        # correctly)
        # Stop issuing when we get to a compute operation
        # While on a compute operation, wait until all pending pim mem requests have finished
        # Then do the compute operation on the tick after they have finished
        self.active_ins_handler()
        self.buf.tick()
        _ = super().tick(cmd)
