from collections import deque
from typing import Any, Callable, Generic, TypeVar, override
from lib.controller.commands import Command, CommandType
from lib.monad import Ptr
from lib.memsys import MemSystem

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


class BaselineState:
    pass


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
    # pim_obj_id -> (base_addr, base_addr + size)
    _pim_objects: dict[int, tuple[int, int]] = field(default_factory=dict)
    
    @classmethod
    def baseline(cls) -> "ControllerState[BaselineState]":
        us = BaselineState()
        # TODO - populate initial controller state
        return ControllerState(user_state=us)


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

    def malloc_obj(self, pim_obj_id: int, addr_range: int, base_addr: int = 0):
        """Register a PIM object's address range with the controller."""
        self.state._pim_objects[pim_obj_id] = (base_addr, base_addr + addr_range)

    def free_obj(self, pim_obj_id: int):
        """Remove a PIM object from the controller's mapping."""
        if pim_obj_id in self.state._pim_objects:
            del self.state._pim_objects[pim_obj_id]

    @override
    def __repr__(self):
        return (
            f"Controller(cycle={self.state._cycle}, "
            f"queue_depth={len(self.state._command_queue)}, "
            f"emit={self.state._emit_command}, "
            f"pim_objects={self.state._pim_objects}, "
            f"user_state={self.state.user_state})"
        )

    def tick(self):
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
            cmd = Command(txn.op,operand_1=txn.id_or_addr)
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
                type=txn.op,
                operand_1=txn.id_addr_base_1,
                operand_2=txn.id_addr_end_1,
                operand_3=txn.id_addr_base_2,
                operand_4=txn.id_addr_end_2,
                scalar=txn.scalar,
                dst_reg=None, # not sure what to do with this
            )
        self.state._command_queue.append(cmd)
    
    @classmethod
    def baseline(cls, mem_pointer: Ptr[MemSystem]) -> "Controller[BaselineState]":
        """Returns a memory controller that operates according to the methods outlined in PIM-ACT"""



        return Controller(
            starting_state=ControllerState.baseline(),
            command_functions=[],
            mem_pointer=mem_pointer,
        )