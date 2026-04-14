"""
Scheduling policy functions for PIM-ACT baseline controller.

Implements T-balancer and L-balancer from:
  "Architecting Compatible PIM Protocol for CPU-PIM Collaboration"
  (Yu et al., IEEE CAL 2024)

These functions are intended to be used as cmd_functions in Controller.baseline().
They read/mutate ControllerState[BaselineState] and return a Command or None.

These functions only handle selection policy. Correctness is checked later on in the function chain
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .controller import ControllerState, BaselineState

from .commands import Command, CommandType

def _is_pim(cmd: Command) -> bool:
    """Return True if the command is a PIM operation (not MEM, not NOP, not mode-switch)."""
    ct = cmd.cmdtype
    if ct.is_mem() or ct == CommandType.NOP:
        return False
    if ct in (CommandType.SWITCH_MODE_PIM, CommandType.SWITCH_MODE_MEM):
        return False
    return True   

def _is_mem(cmd: Command) -> bool:
    return cmd.cmdtype.is_mem()


# ---------------------------------------------------------------------------
# Mode-switch injection
# ---------------------------------------------------------------------------

def mode_switch_gate(state: ControllerState[BaselineState]) -> Command | None:
    """
    If a mode-switch command was queued by the scheduling policy on the previous
    tick, emit it now and clear the pending flag.

    If a switch is pending, it takes priority over everything. 
    """
    us = state.user_state

    if us.mode_switch_pending is not None:
        cmd = us.mode_switch_pending
        us.mode_switch_pending = None
        # Update DRAM mode
        if cmd.cmdtype == CommandType.SWITCH_MODE_PIM:
            us.current_dram_mode = True
        elif cmd.cmdtype == CommandType.SWITCH_MODE_MEM:
            us.current_dram_mode = False
        # This command is NOT in _command_queue, so we add it and let tick() remove it.
        state._command_queue.appendleft(cmd)
        return cmd

    return None


# ---------------------------------------------------------------------------
# T-balancer
# ---------------------------------------------------------------------------

def fr_fcfs(state: ControllerState[BaselineState],
                          previous_check: Callable[[Command], bool]) -> Command | None:
    """
    TODO - need to be tracking open rows for FR-FCFS, which we don't

    currently just does FCFS
    """
    for cmd in state._command_queue:
        if previous_check(cmd):
            return cmd # select the first operation that shares a type with the previous operation 
    return None


def t_balancer(state: ControllerState[BaselineState]) -> Command | None:
    """
    Throughput-balanced scheduling (T-balancer).

    Limits consecutive requests of the same type to "threshold".
    When the limit is hit or no more requests of the prioritized type exist,
    switch priority to the other type.

    Within a type, selection follows FR-FCFS
    """
    us = state.user_state

    if not state._command_queue:
        return None

    # Determine which predicate to use based on current priority
    if us.currently_is_pim:
        primary_pred = _is_pim
        secondary_pred = _is_mem
    else:
        primary_pred = _is_mem
        secondary_pred = _is_pim

    # Select the next available of the same type using FR_FCFS
    selected = fr_fcfs(state, primary_pred)

    if selected is not None and us.consecutive_count < us.threshold:
        us.consecutive_count += 1
    else:
        # Threshold exceeded or no primary-type commands — switch
        us.currently_is_pim = not us.currently_is_pim
        us.consecutive_count = 1
        selected = fr_fcfs(state,
            secondary_pred if selected is None else (
                _is_pim if us.currently_is_pim else _is_mem
            ),
        )
        if selected is None:
            # Nothing of the switched type either — try first in queue
            selected = state._command_queue[0]
            if selected is not None:
                us.currently_is_pim = _is_pim(selected)
                us.consecutive_count = 1

    if selected is None:
        return None

    return _inject_switch_if_needed(state, selected)


def l_balancer(state: ControllerState[BaselineState]) -> Command | None:
    """
    Latency-balanced scheduling (L-balancer).

    Prioritizes PIM requests. If the longest-pending MEM request's
    queuing latency exceeds "threshold" cycles, switch to draining ALL pending
    normal requests before returning to PIM.

    Within each type, selection follows FR-FCFS
    """
    us = state.user_state

    if not state._command_queue:
        us.draining_mem = False # stop draining command queue if the queue is empty
        return None

    # Check if any normal request has waited too long, or whether we need to switch types
    if not us.draining_mem:
        if us.earliest_mem is None: # if there is no oldest mem
            for cmd in state._command_queue: # stop at the first mem - in a queue, the oldest is the first.
                if _is_mem(cmd): 
                    us.earliest_mem = cmd
                    break
        elif state._cycle - us.earliest_mem.entry_time  >= us.threshold: # check for whether we should start draining
            us.draining_mem = True
            us.earliest_mem = None

    if us.draining_mem:
        # Drain mem requests
        selected = fr_fcfs(state, _is_mem)
        if selected is None:
            # All mem requests drained, back to PIM
            us.draining_mem = False
            selected = fr_fcfs(state, _is_pim)
    else:
        # Default: prioritize PIM
        selected = fr_fcfs(state, _is_pim)
        if selected is None:
            selected = fr_fcfs(state, _is_mem)
    
        if selected is not None:
            us.earliest_mem = None
            us.draining_mem = True

    if selected is None:
        return None

    return _inject_switch_if_needed(state, selected)


def _inject_switch_if_needed(state: ControllerState[BaselineState],
                          selected: Command) -> Command | None:
    """
    If "selected" requires a DRAM mode different from the current one,
    create a mode-switch command, stash it as pending, stash the current command,
    and set the current enqueued command to none.

    If no switch is needed, return None
    """
    # for NOP (or other commands that dont care about PIM or MEM mode)
    if not _is_pim(selected) and not _is_mem(selected):
        return selected
    
    us = state.user_state
    required_mode = _is_pim(selected)

    if required_mode == us.current_dram_mode:
        return selected

    # Need to switch. Synthesize the switch command.
    if required_mode:
        switch_cmd = Command(state._cycle,type=CommandType.SWITCH_MODE_PIM)
        # Don't emit the data command this tick
        # The data command stays in the queue and will (probably) be selected next due to FR_FCFS
        state._emit_command = None
    else:
        switch_cmd = Command(state._cycle,type=CommandType.SWITCH_MODE_MEM)
        # Don't emit the data command this tick
        # The data command stays in the queue and will (probably) be selected next due to FR_FCFS
        state._emit_command = None
    us.mode_switch_pending = switch_cmd
    return None

# selects between T-balancer or L-balancer
def scheduling_policy(state: ControllerState[BaselineState]) -> Command | None:
    """
    select between T-balancer or L-balancer based on pim_mode.
    """
    if state.user_state.pim_mode:
        return t_balancer(state)
    else:
        return l_balancer(state)