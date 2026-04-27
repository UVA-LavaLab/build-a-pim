from collections import deque
from configparser import ConfigParser
from typing import Any, Callable, Generic, TypeVar, override

from lib.controller.response import Response
from .commands import Command, CommandType
from ..monad import Ptr
from ..memsys import MemSystem
from .correctness import correctness, printMemConfig, update_timing_state
from .scheduling import mode_switch_gate, scheduling_policy

# dataclass auto-generates __init__, __repr__, and __eq__
from dataclasses import dataclass, field


@dataclass
class Transaction:
    op: CommandType 
    id_or_addr: int # ID of PIM or address of MEM
    id_addr_base_1: int
    id_addr_end_1: int
    id_addr_base_2: int
    id_addr_end_2: int
    id_dst: int
    scalar: Any


@dataclass
class BaselineState:
    num_requests_in_current_mode: int = 0
    time_spent_in_current_mode: int = 0
    number_of_mem_requests: int = 0
    number_of_pim_requests: int = 0
 
    # Scheduling bookkeeping
    current_dram_mode: bool = False          # True = PIM mode, False = MEM mode
    currently_is_pim: bool = False            # Which type currently has scheduling priority
    consecutive_count: int = 0               # T-balancer: consecutive same-type commands issued
    draining_mem: bool = False            # L-balancer: currently draining all MEM requests
    earliest_mem: Command | None = None  # L-balancer: keeps track of the oldest MEM request in the queue
    mode_switch_pending: Command | None = None  # Synthesized switch cmd awaiting emission

    enqueue_cycles: dict[int, int] = field(default_factory=dict)  # command id -> cycle entered queue  


 
    def __init__(self, pim_mode: bool = True, threshold: int = 32) -> None:
        self.pim_mode: bool = pim_mode       # True = T-balancer, False = L-balancer
        self.threshold: int = threshold


@dataclass
class ControllerState[T]:
    """
    Per-cycle mutable state passed to command selection functions.

    T is a user-defined type holding custom state. The
    controller owns this object and passes it to each cmd_function on every tick.

    Fields managed by Controller (anything outside of user_state and pass_memory_transactions))
    should generally not be modified by cmd_functions
    """

    # user-managed
    user_state: T
    pass_memory_transactions: bool = True # Determines whether mem transactions should pass through the controller.

    # controller-managed
    _command_queue: deque[Command] = field(default_factory=deque)
    _emit_command: Command | None = None
    _cycle: int = 0
    _act_timestamps: list[int] = field(default_factory=list)
    
    # To be used by correctness.py
    _last_act_cycle: int = 0
    _last_pre_cycle: int = 0
    _last_rd_cycle: int = 0
    _last_wr_cycle: int = 0
    _last_cas_cycle: int = 0

    # pim_obj_id -> (base_addr, base_addr + size)
    _pim_objects: dict[int, tuple[int, int]] = field(default_factory=dict)

    # read only memory configuration
    _mem_config: ConfigParser = field(default_factory=ConfigParser)


    def parse_memory_config(self, conf_file:str):
        _ = self._mem_config.read(conf_file)

    
    @classmethod
    def baseline(cls, conf_file:str, mode:bool, threshold:int) -> "ControllerState[BaselineState]":
        """
        Returns an Initial State usable by the memory controller for PIM-ACT
        
        conf_file is the path to the configuration file to be parsed for memory timings

        mode determines how arbitration between PIM and MEM requests is done.
        mode = True: Throughput-balanced mode
        mode = False: Latency-balanced mode

        threshold determines the throughput or latency limits before requiring a PIM-ACT switch.
        """
        us = BaselineState(mode, threshold)
    
        retval = ControllerState(user_state=us)

        retval.parse_memory_config(conf_file)

        return retval
    


class Controller[T]:
    """
    Schedules and emits commands to a PIM device.

    Generic over T, a user-defined state type. Users customize scheduling
    behavior by providing:
    1. A starting ControllerState[T] with their custom state in user_state
    2. A list of cmd_functions that read/mutate ControllerState[T] and
       return a Command from ControllerState.command_queue (or None to abstain)

    On each tick(), cmd_functions are called in order. Later functions can
    override earlier selections. The last non-None return becomes the
    emitted command.

    Currently, the class only supports sending a single command per tick.
    """

    def __init__(
        self,
        starting_state: ControllerState[T],
        command_functions: list[Callable[[ControllerState[T]], Command | None]],
        mem_pointer: Ptr[MemSystem],
    ):
        self.cmd_functions: list[Callable[[ControllerState[T]], Command | None]] = (
            command_functions
        )
        self.command_queue: deque[Command] = deque()
        self.p_mem: Ptr[MemSystem] = mem_pointer
        self.state: ControllerState[T] = starting_state
        self.responses: list[Response] = []

    def malloc_obj(self, pim_obj_id: int, addr_range: int, base_addr: int = 0):
        """Register a PIM object's address range with the controller."""
        self.state._pim_objects[pim_obj_id] = (base_addr, base_addr + addr_range)

    def free_obj(self, pim_obj_id: int):
        """Remove a PIM object from the controller's mapping."""
        if pim_obj_id in self.state._pim_objects:
            del self.state._pim_objects[pim_obj_id]

    def push_response(self, response: Response):
        self.responses.append(response)

    @override
    def __repr__(self):
        return (
            f"Controller(cycle={self.state._cycle}, "
            f"queue_depth={len(self.state._command_queue)}, "
            f"emit={self.state._emit_command}, "
            f"pim_objects={self.state._pim_objects}, "
            f"user_state={self.state.user_state})"
        )

    def tick(self, trans: Transaction | None = None):
        """
        The core of the Controller class. Selects a Command to emit for the current cycle

        cmd_functions are called in order. Later functions have higher priority
        (can overwrite earlier selections), but earlier functions can mutate
        controller state first.

        emit_command is reset to None at the start of each tick.
        """
        self.state._cycle += 1
        self.state._emit_command = None

        for f in self.cmd_functions:
            cmd = f(self.state)
            if cmd is not None:
                self.state._emit_command = cmd

        if self.state._emit_command is not None:
            self.state._command_queue.remove(self.state._emit_command)

        return self.state._emit_command

    def push_transaction(self, txn: Transaction):

        cmd = None
        if txn.op.is_mem() and self.state.pass_memory_transactions:
            _ = self.p_mem().add_transaction(
                txn.id_or_addr, txn.op == CommandType.MEM_WRITE, False
            )
            return
        elif txn.op.is_mem():
            cmd = Command(self.state._cycle,txn.op,operand_1=txn.id_or_addr)
        elif txn.op == CommandType.PIM_MALLOC:
            # txn.id_or_addr = pim object ID, txn.id_op2 = size
            # TODO: base_addr assignment — needs allocator or explicit arg
            self.malloc_obj(txn.id_or_addr, txn.id_addr_base_1, txn.id_addr_end_1)
            return

        elif txn.op == CommandType.PIM_FREE:
            self.free_obj(txn.id_or_addr)
            return
        else:
            cmd = Command(
                self.state._cycle,
                type=txn.op,
                operand_1=txn.id_addr_base_1,
                operand_2=txn.id_addr_end_1,
                operand_3=txn.id_addr_base_2,
                operand_4=txn.id_addr_end_2,
                dst_reg=None, # not sure what to do with this
            )
        self.state._command_queue.append(cmd)
    
    @classmethod
    def baseline(cls, mem_pointer: Ptr[MemSystem], conf_file:str, mode:bool = True, threshold:int = 10) -> "Controller[BaselineState]":
        """
        Returns a memory controller implementing PIM-ACT scheduling.
    
        mode = True  - T-balancer (throughput-balanced)
        mode = False - L-balancer (latency-balanced)
        threshold    - the switching limit for the chosen policy
    
        Command function chain (in priority order, later overrides earlier):
        1) mode_switch_gate      — emits pending SWITCH_MODE_* if one was queued
        2) scheduling_policy     — picks the data command (T-balancer or L-balancer)
        3) update_timing_state  — timing bookkeeping
        4) correctness           — enforces DRAM timing legality

        """
        return Controller(
            starting_state=ControllerState.baseline(conf_file, mode, threshold),
            command_functions=[
                scheduling_policy,
                mode_switch_gate,
                update_timing_state,
                correctness,
                #printMemConfig,
            ],
            mem_pointer=mem_pointer,
        )
    
# testbed (for now)
if __name__ == "__main__":
    config = ""
    memsys = MemSystem(config,".")
    pointer = Ptr[MemSystem](memsys)
    control = Controller.baseline(pointer,config)
    control.tick()
