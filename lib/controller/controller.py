from collections import deque
from typing import Any, Callable
from lib.controller.commands import Command

from configparser import ConfigParser

from dataclasses import (
    dataclass,
)  # dataclass auto-generates __init__, __repr__, and __eq__


# Represents input coming in from CPU
@dataclass
class InputClass:
    op: int
    addr: int


# stores timing & interconnect info
class MemoryConfig:
    def __init__(self, conf_file: str, control_bandwidth: int):
        self.config: ConfigParser = ConfigParser()
        _ = self.config.read(conf_file)
        self.control_bandwidth: int = control_bandwidth


# unused for now
@dataclass
class ControllerConfig:
    pass


class ControllerState:
    def __init__(self):
        self.cycle: int = 0
        self.act_timestamps: list[int] = []


class Controller:
    def __init__(
        self,
        cmd_set: list[Any],
        memConfig: MemoryConfig,
        controlConfig: ControllerConfig | None = None,
    ):
        self.command_set: list[Any] = cmd_set
        self.cmd_functions: list[Callable[[ControllerState], Command | None]] = []
        self.emit_functions: list[Callable[[ControllerState], bool]] = []
        self.command_queue: deque[Command] = deque()

        self.state: ControllerState = ControllerState()

        self.memory_config: MemoryConfig = (
            memConfig  # stores timing & interconnect info
        )
        if not controlConfig is None:
            self.pim_config: ControllerConfig = (
                controlConfig  # stores any additional info about the Controller
            )
        else:
            self.pim_config: ControllerConfig = ControllerConfig()

    def __repr__(self):
        pass  # TODO

    def tick(self):
        for f in self.cmd_functions:
            cmd = f(self.state)
            if cmd is not None:
                # TODO: determine when to prepend the command
                # also determine if a PQ is useful
                self.command_queue.append(cmd)

        for f in self.emit_functions:
            if not f(self.state):
                return None

        return self.command_queue.popleft()

    def input(self, input: InputClass):
        pass
        # TODO: Take generic input from the CPU to be translated into a
        # PIM command and put into the command_queue


if __name__ == "__main__":
    test1 = InputClass(op=5, addr=10)
    print(test1.op)
    print(test1.addr)

    test2 = MemoryConfig(name="memory1", timings={"tCCD": 500}, controlBandwidth=1)
    print(test2)
    print(test2.timings)

    test3 = Controller(123, test2)
