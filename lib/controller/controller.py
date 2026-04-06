from collections import deque
from typing import Any, Callable, Generic, TypeVar
from lib.controller.commands import Command
from monad import Ptr
from configparser import ConfigParser


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
class InputClass:
    op: int
    addr: int

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
    ):
        # --- Selection & Emission ---
        self.cmd_functions: list[Callable[[ControllerState[T]], Command | None]] = command_functions
        self.command_queue: deque[Command] = deque()

        # --- Internal State ---
        self.state : ControllerState[T] = starting_state


    def __repr__(self):
        pass  # TODO


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
    # Th
    # The first one to output a value 
    def input(self, input: InputClass):
        # TODO: Change the Command constructor here to match the actual Command
        command: Command = Command(input) # placeholder construction
        self.state.command_queue.append(command)
