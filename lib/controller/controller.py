from collections import deque
from typing import Any, Callable, Generic, TypeVar, override
from lib.controller.commands import Command, CommandType
from lib.monad import Ptr
from lib.memsys import MemSystem


from dataclasses import (
    dataclass,
)  # dataclass auto-generates __init__, __repr__, and __eq__

"""
TODO:

Moved all of the validation/timing burden into Device. 
With the amount of reconfigurability here, ideally the user would never have to modify it.

The next steps would be to:

1) create a set of rules that obey PIM-ACT and implement them as a baseline controller config
2) Expand InputClass and input() to support the needs of the Command class
"""


# Represents input coming in from CPU
@dataclass
class Transaction:
    op: CommandType # I think we can just make this use CommandType, since there would be 100% overlap with any hypothetical TransactionType
    # this could be the ID of the associated PIM object
    # OR the address of the corresponding mem transaction
    id_or_addr: int
    # passed IDs of pim objects (in the event of a second operator)
    # if you can figure out a better way to do this, go for it
    id_op2: int
    id_dst: int
    # we also need to be able to store scalars of any type in case users want to add / scale / whatever by a scalar
    # this is any type because if we restrict to int or float, that ruins compatibility with numpy
    scalar: Any


@dataclass
class ControllerState[T]:
    """
    Per-cycle mutable state passed to command selection functions.

    T is a user-defined type holding custom state. The
    controller owns this object and passes it to each cmd_function on every tick.

    Fields managed by Controller (anything outside of user_state)
    should generally not be modified by cmd_functions
    """

    user_state: T
    command_queue: deque[Command]
    emit_command: Command | None = None
    cycle: int = 0
    act_timestamps: list[int] = []
    pim_objects: dict[int, tuple[int, int]] = {}
    
    # TODO: should modify the pim_objects dictionary to define which address 
    # ranges correspond to a particular pim object.
    # bounds-checking is tempting, but there can be overlapping pim objects
    # with some implementations, so we should generally avoid that

    # This also might warrant a separate PimObj type to mimic pimeval, but 
    # the type conversion could also be a waste of time

    # This functionality is also not realistic, since this sort of this would
    # likely be held in the TLB, but this is the closest we can get to the TLB 
    # without writing a CPU simulator. Also, a lot of PIM proposals have their own
    # TLB anyways, so it would be fair to just use that as a justification for why the MC
    # may have its own memory mapping and argue that this is just a modeling for that
    def malloc_obj(self, pim_obj_id: int, addr_range: int):
        pass

    # remove the object from the memory controller's mapping
    def free_obj(self, pim_obj_id: int):
        pass


class Controller[T]:
    """
    Schedules and emits commands to a PIM device.

    Generic over T, a user-defined state type. Users customize scheduling
    behavior by providing:
    1. A starting ControllerState[T] with their custom state in user_state
    2. A list of cmd_functions that read/mutate ControllerState[T] and
       return a Command from ControlerState.command_queue (or None to abstain)

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
        # --- Selection & Emission ---
        self.cmd_functions: list[Callable[[ControllerState[T]], Command | None]] = (
            command_functions
        )
        self.command_queue: deque[Command] = deque()
        self.p_mem: Ptr[MemSystem] = mem_pointer

        # --- Internal State ---
        self.state: ControllerState[T] = starting_state

    @override
    def __repr__(self):
        # TODO: formalize (updated to avoid LSP complaints)
        return ""

    def tick(self):
        """
        The core of the Controller class. Selects a Command to emit for the current cycle

        self.cmd_functions are called in the order given to select a Command. Later functions have higher
        priority (are able to overwrite previous functions), but earlier functions can mutate
        controller state first.

        self.emit_command holds the current command to be emitted. It is set to none at the beginning of each tick
        """
        self.state.cycle += 1
        self.state.emit_command = None

        for f in self.cmd_functions:
            cmd = f(self.state)
            self.state.emit_command = cmd

        if self.state.emit_command != None:
            self.state.command_queue.remove(self.state.emit_command)

        return self.state.emit_command

    # input is directly run through the set of command functions, which are given access to the current command queue.
    # The first one to output a value
    # NOTE: renamed to "push_transaction" to be clearer about functionality and not reuse the input keyword
    def push_transaction(self, txn: Transaction):
        # TODO: Change the Command constructor here to fill out the other fields of the command
        command: Command = Command(txn.op)  # placeholder construction
        if command.is_mem():
            _ = self.p_mem().add_transaction(
                command.addr, command.cmdtype == CommandType.MEM_WRITE, False
            )
        elif command.is_malloc():
            # TODO: placeholders, flesh out later
            self.state.malloc_obj(0, 0)
        elif command.is_free():
            # TODO: placeholders, flesh out later
            self.state.free_obj(0)
        else:
            self.state.command_queue.append(command)
