from unittest.mock import MagicMock
from lib.controller.controller import Controller, ControllerState, Transaction, BaselineState
from lib.controller.commands import Command, CommandType
from lib.containers import Ptr

# Note: These test cases were AI-generated then cleaned up by hand
def make_mock_mem() -> Ptr:
    """Returns a Ptr wrapping a mock MemSystem."""
    mem = MagicMock()
    mem.add_transaction = MagicMock(return_value=True)
    return Ptr(mem)


def make_controller(cmd_functions=None, user_state=None):
    """Build a Controller with sensible defaults for testing."""
    if user_state is None:
        user_state = BaselineState()
    state = ControllerState(user_state=user_state)
    return Controller(
        starting_state=state,
        command_functions=cmd_functions or [],
        mem_pointer=make_mock_mem(),
    )


# --- tick() ---

class TestTick:
    def test_tick_increments_cycle(self):
        ctrl = make_controller()
        assert ctrl.state._cycle == 0
        ctrl.tick()
        assert ctrl.state._cycle == 1
        ctrl.tick()
        assert ctrl.state._cycle == 2

    def test_tick_no_functions_emits_none(self):
        ctrl = make_controller()
        result = ctrl.tick()
        assert result is None
        assert ctrl.state._emit_command is None

    def test_tick_single_function_returns_command(self):
        cmd = Command(CommandType.PIM_ADD)
        ctrl = make_controller()
        ctrl.state._command_queue.append(cmd)

        def select_first(state):
            if state._command_queue:
                return state._command_queue[0]
            return None

        ctrl.cmd_functions = [select_first]
        result = ctrl.tick()
        assert result is cmd
        assert cmd not in ctrl.state._command_queue

    def test_tick_later_function_overrides_earlier(self):
        cmd_a = Command(CommandType.PIM_ADD)
        cmd_b = Command(CommandType.PIM_SUB)
        ctrl = make_controller()
        ctrl.state._command_queue.extend([cmd_a, cmd_b])

        def pick_a(state):
            return cmd_a

        def pick_b(state):
            return cmd_b

        ctrl.cmd_functions = [pick_a, pick_b]
        result = ctrl.tick()
        assert result is cmd_b
        assert cmd_b not in ctrl.state._command_queue
        assert cmd_a in ctrl.state._command_queue

    def test_tick_later_function_returning_none_does_not_override(self):
        cmd = Command(CommandType.PIM_ADD)
        ctrl = make_controller()
        ctrl.state._command_queue.append(cmd)

        def pick(state):
            return cmd

        def abstain(state):
            return None

        ctrl.cmd_functions = [pick, abstain]
        result = ctrl.tick()
        assert result is cmd

    def test_tick_removes_emitted_command_from_queue(self):
        cmd = Command(CommandType.PIM_ADD)
        ctrl = make_controller()
        ctrl.state._command_queue.append(cmd)
        ctrl.cmd_functions = [lambda s: cmd]
        ctrl.tick()
        assert len(ctrl.state._command_queue) == 0

    def test_tick_function_can_mutate_state(self):

        class TempState:
            def __init__(self):
                self.touched = False
        
        ctrl = make_controller(user_state=TempState())

        def mutator(state):
            state.user_state.touched = True
            return None

        ctrl.state.user_state.touched = False
        ctrl.cmd_functions = [mutator]
        ctrl.tick()
        assert ctrl.state.user_state.touched is True


# --- push_transaction() ---

class TestPushTransaction:
    def test_mem_read_passthrough(self):
        ctrl = make_controller()
        ctrl.state.pass_memory_transactions = True
        txn = Transaction(
            op=CommandType.MEM_READ,
            id_or_addr=0x1000,
            id_addr_base_1=0, id_addr_end_1=0,
            id_addr_base_2=0, id_addr_end_2=0,
            id_dst=0, scalar=None,
        )
        ctrl.push_transaction(txn)
        ctrl.p_mem().add_transaction.assert_called_once_with(0x1000, False, False)
        assert len(ctrl.state._command_queue) == 0

    def test_mem_write_passthrough(self):
        ctrl = make_controller()
        txn = Transaction(
            op=CommandType.MEM_WRITE,
            id_or_addr=0x2000,
            id_addr_base_1=0, id_addr_end_1=0,
            id_addr_base_2=0, id_addr_end_2=0,
            id_dst=0, scalar=None,
        )
        ctrl.push_transaction(txn)
        ctrl.p_mem().add_transaction.assert_called_once_with(0x2000, True, False)

    def test_mem_no_passthrough_enqueues(self):
        ctrl = make_controller()
        ctrl.state.pass_memory_transactions = False
        txn = Transaction(
            op=CommandType.MEM_READ,
            id_or_addr=0x1000,
            id_addr_base_1=0, id_addr_end_1=0,
            id_addr_base_2=0, id_addr_end_2=0,
            id_dst=0, scalar=None,
        )
        ctrl.push_transaction(txn)
        ctrl.p_mem().add_transaction.assert_not_called()
        assert len(ctrl.state._command_queue) == 1

    def test_malloc_registers_object(self):
        ctrl = make_controller()
        txn = Transaction(
            op=CommandType.PIM_MALLOC,
            id_or_addr=7,
            id_addr_base_1=0x100, id_addr_end_1=0x200,
            id_addr_base_2=0, id_addr_end_2=0,
            id_dst=0, scalar=None,
        )
        ctrl.push_transaction(txn)
        assert 7 in ctrl.state._pim_objects
        assert len(ctrl.state._command_queue) == 0

    def test_free_removes_object(self):
        ctrl = make_controller()
        ctrl.state._pim_objects[7] = (0x100, 0x200)
        txn = Transaction(
            op=CommandType.PIM_FREE,
            id_or_addr=7,
            id_addr_base_1=0, id_addr_end_1=0,
            id_addr_base_2=0, id_addr_end_2=0,
            id_dst=0, scalar=None,
        )
        ctrl.push_transaction(txn)
        assert 7 not in ctrl.state._pim_objects

    def test_free_nonexistent_is_noop(self):
        ctrl = make_controller()
        txn = Transaction(
            op=CommandType.PIM_FREE,
            id_or_addr=99,
            id_addr_base_1=0, id_addr_end_1=0,
            id_addr_base_2=0, id_addr_end_2=0,
            id_dst=0, scalar=None,
        )
        ctrl.push_transaction(txn)  # should not raise

    def test_pim_compute_enqueues_command(self):
        ctrl = make_controller()
        txn = Transaction(
            op=CommandType.PIM_ADD,
            id_or_addr=0,
            id_addr_base_1=0x100, id_addr_end_1=0x200,
            id_addr_base_2=0x300, id_addr_end_2=0x400,
            id_dst=0, scalar=2.0,
        )
        ctrl.push_transaction(txn)
        assert len(ctrl.state._command_queue) == 1
        cmd = ctrl.state._command_queue[0]
        assert cmd.cmdtype == CommandType.PIM_ADD
        assert cmd.range_1 == (0x100, 0x200)
        assert cmd.range_2 == (0x300, 0x400)


# --- malloc_obj / free_obj ---

class TestMallocFree:
    def test_malloc_stores_range(self):
        ctrl = make_controller()
        ctrl.malloc_obj(1, 256, base_addr=0x1000)
        assert ctrl.state._pim_objects[1] == (0x1000, 0x1000 + 256)

    def test_malloc_overwrites_existing(self):
        ctrl = make_controller()
        ctrl.malloc_obj(1, 100, base_addr=0)
        ctrl.malloc_obj(1, 200, base_addr=0x500)
        assert ctrl.state._pim_objects[1] == (0x500, 0x500 + 200)

    def test_free_deletes(self):
        ctrl = make_controller()
        ctrl.malloc_obj(1, 100)
        ctrl.free_obj(1)
        assert 1 not in ctrl.state._pim_objects


# --- Integration: push then tick ---

class TestIntegration:
    def test_push_then_tick_fifo(self):
        def fifo(state):
            if state._command_queue:
                return state._command_queue[0]
            return None

        ctrl = make_controller(cmd_functions=[fifo])
        for op in [CommandType.PIM_ADD, CommandType.PIM_SUB, CommandType.PIM_MUL]:
            ctrl.push_transaction(Transaction(
                op=op,
                id_or_addr=0,
                id_addr_base_1=0, id_addr_end_1=0,
                id_addr_base_2=0, id_addr_end_2=0,
                id_dst=0, scalar=None,
            ))

        assert len(ctrl.state._command_queue) == 3
        assert ctrl.tick().cmdtype == CommandType.PIM_ADD
        assert ctrl.tick().cmdtype == CommandType.PIM_SUB
        assert ctrl.tick().cmdtype == CommandType.PIM_MUL
        assert ctrl.tick() is None
