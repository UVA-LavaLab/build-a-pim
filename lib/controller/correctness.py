from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .controller import ControllerState
from .commands import Command, CommandType

def _check_act_timing(state: "ControllerState", cycle: int) -> bool:
    """Check timing constraints for commands that involve row activation."""
    tRP = state._mem_config.getint("timing", "tRP")
    tRRD_S = state._mem_config.getint("timing", "tRRD_S")
    tFAW = state._mem_config.getint("timing", "tFAW")

    return True


def _check_cas_timing(state: "ControllerState", cycle: int) -> bool:
    """Check timing constraints for commands that involve CAS (column read/write)."""
    tCCD_S = state._mem_config.getint("timing", "tCCD_S")
    tRCD = state._mem_config.getint("timing", "tRCD")

    return True


def _check_pre_timing(state: "ControllerState", cycle: int) -> bool:
    """Check timing constraints for precharge commands."""
    tRAS = state._mem_config.getint("timing", "tRAS")
    tWR = state._mem_config.getint("timing", "tWR")
    tRTP = state._mem_config.getint("timing", "tRTP")

    return True


def update_timing_state(state: "ControllerState") -> Command | None:
    """Update timing tracking after a command is approved."""
    return None

def printMemConfig(state: "ControllerState") -> Command | None:
    sections = state._mem_config.sections()
    for section in sections:
        print("-------------------------------")
        print("Section: ", section)
        print(state._mem_config.items(section))
        print("-------------------------------")
    return None

def correctness(state: "ControllerState") -> Command | None:
    """
    Timing correctness check. If the current _emit_command violates any
    DRAM timing constraint, it is replaced with None.

    Planned constraints to enforce:
        tRCD  — activate -> read/write 
        tRP   — precharge -> activate (same bank)
        tRAS  — activate -> precharge (minimum row open time)
        tWR   — last write data -> precharge
        tRTP  — read -> precharge
        tRRD_S — activate -> activate (different bank groups)
        tFAW  — four-activate window
        tCCD_S — CAS -> CAS (different bank groups)
    """
    cmd = state._emit_command
    if cmd is None:
        return None

    cycle = state._cycle
    ok = True

    ok = ok and ( _check_act_timing(state, cycle) 
            and _check_cas_timing(state, cycle)
            and _check_pre_timing(state, cycle))

    if not ok:
        state._emit_command = None

    return None
