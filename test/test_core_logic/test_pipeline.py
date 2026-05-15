from lib.cores.components.pipeline import Pipeline
from lib.cores.instructions import IState as IS, OpType as OT
from lib.memsys import MemSystem
from lib.containers import Ptr, Box
from lib.cores.ins_stream_bank_simd import Core as SC
import numpy as np
import numpy.typing as npt
import math


def dev_setup() -> tuple[MemSystem, SC, Pipeline]:
    mem = MemSystem("./dramsim3/configs/HBM2_8Gb_x128.ini", ".", nd_log=True)
    core = SC((0, 0, 0, 0), Ptr(mem))
    return (mem, core, core.pipeline)


def list_setup() -> (
    tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32]]
):
    base = np.zeros(128, dtype=np.int32) + 1
    asc = np.arange(128, dtype=np.int32)
    desc = np.arange(0, 128, step=-1, dtype=np.int32)

    return (base, asc, desc)


def map_list_at_0(mem: MemSystem, base: npt.NDArray[np.int32]):
    id = mem.add_data_structure(base)

    base_bytes = np.frombuffer(base, dtype=np.uint8)
    gdl_bytes: int = int(mem.m_gdl_width / 8)
    len_chunks: int = int(math.ceil(len(base_bytes) / gdl_bytes))

    mem.mmap(0, 0, 0, 0, 0, id, len_chunks, offset=0)


def test_pipeline_read_starts_at_mem_stage():
    mem, core, pipe = dev_setup()
    base, _, _ = list_setup()

    map_list_at_0(mem, base)

    core.add_instruction(OT.READ, addr=0x0, dst="vrA")
    while pipe.stages[-3].ins is None:
        core.tick()
        mem.tick()

    # read should be in the execute stage and should NOT be started
    assert pipe.stages[-3].ins is not None
    assert pipe.stages[-3].ins.operation == OT.READ
    assert pipe.stages[-3].ins.state == IS.COLD

    core.tick()
    mem.tick()

    # now, read should be in the mem stage and should be started
    assert pipe.stages[-2].ins is not None
    assert pipe.stages[-2].ins.operation == OT.READ
    assert pipe.stages[-2].ins.state == IS.WARM


def test_pipeline_write_commits_at_end_of_mem_stage():
    mem, core, pipe = dev_setup()
    base, asc, _ = list_setup()

    map_list_at_0(mem, base)

    dw = Box(np.frombuffer(asc, count=int(mem.m_gdl_width / 32)))

    core.set_reg("vrA", dw)
    core.add_instruction(OT.WRITE, addr=0x0, in_reg1="vrA")
    while pipe.stages[-3].ins is None:
        core.tick()
        mem.tick()

    # write should be in the execute stage and should NOT be started
    assert pipe.stages[-3].ins is not None
    assert pipe.stages[-3].ins.operation == OT.WRITE
    assert pipe.stages[-3].ins.state == IS.COLD

    core.tick()
    mem.tick()

    length = int(mem.m_gdl_width / 32)

    # now, mem should be in the mem stage and should be started
    assert pipe.stages[-2].ins is not None
    assert pipe.stages[-2].ins.operation == OT.WRITE
    assert pipe.stages[-2].ins.state == IS.WARM
    # data should be unmodified because the instruction has just arrived at the
    # mem stage
    assert np.all(
        mem.stored_data_structures[0].data_structure[:length] == base[:length]
    )

    # on each cycle, the data in memory should be equal to its unmodified state
    while pipe.stages[-1].ins is None:
        assert np.all(
            mem.stored_data_structures[0].data_structure[:length] == base[:length]
        )
        core.tick()
        mem.tick()

    # mem operation should be done and should be in the writeback stage now,
    # data should have committed
    assert pipe.stages[-1].ins is not None
    assert pipe.stages[-1].ins.operation == OT.WRITE
    assert pipe.stages[-1].ins.state == IS.DONE

    assert np.all(mem.stored_data_structures[0].data_structure[:length] == asc[:length])


def test_pipeline_read_and_write_offsets():
    mem, core, pipe = dev_setup()
    _, asc, _ = list_setup()

    map_list_at_0(mem, asc)

    core.set_reg("rA", 1)
    core.set_reg("rB", 2)
    core.add_instruction(OT.READ, addr=0x0, dst="vrA", in_reg1="rA")
    core.add_instruction(OT.WRITE, in_reg1="vrA", in_reg2="rB", addr=0x0)

    while pipe.stages[-2].ins is None:
        core.tick()
        mem.tick()

    assert pipe.stages[-2].ins is not None
    assert pipe.stages[-2].ins.operation == OT.READ
    assert pipe.stages[-2].ins.addr == 0x1

    while pipe.stages[-2].ins is not None and pipe.stages[-2].ins.operation == OT.READ:
        core.tick()
        mem.tick()

    # now tick once to write back the data to the register file
    core.tick()
    mem.tick()

    length = int(mem.m_gdl_width / 32)
    assert np.all(core.get_reg("vrA").data == asc[length : 2 * length])

    while pipe.stages[-2].ins is None:
        core.tick()
        mem.tick()

    assert pipe.stages[-2].ins is not None
    assert pipe.stages[-2].ins.operation == OT.WRITE
    assert pipe.stages[-2].ins.addr == 0x2

    while pipe.stages[-1].ins is None:
        assert np.all(core.get_reg("vrA").data == asc[length:2*length])
        core.tick()
        mem.tick()

    assert pipe.stages[-1].ins is not None
    assert pipe.stages[-1].ins.operation == OT.WRITE
    assert pipe.stages[-1].ins.state == IS.DONE
    assert np.all(
        mem.stored_data_structures[0].data_structure[2 * length : 3 * length]
        == asc[length : 2 * length]
    )


def test_pipeline_vec_add_timing_and_register_states():
    mem, core, pipe = dev_setup()
    base, _, _ = list_setup()

    map_list_at_0(mem, base)

    core.add_instruction(OT.READ, addr=0x0, dst="vrA")
    core.add_instruction(OT.READ, addr=0x1)
    core.add_instruction(OT.VEC_ADD, in_reg1="vrA", in_reg2="gdl")

    while pipe.stages[-3].ins is None or pipe.stages[-3].ins.operation == OT.READ:
        core.tick()
        mem.tick()

    # vec_add should be in the execute stage and should be started
    assert pipe.stages[-3].ins is not None
    assert pipe.stages[-3].ins.operation == OT.VEC_ADD
    assert pipe.stages[-3].ins.state == IS.WARM

    core.tick()
    mem.tick()

    length = int(mem.m_gdl_width / 32)

    # now, vec_add should be in the mem stage and should be complete
    assert pipe.stages[-2].ins is not None
    assert pipe.stages[-2].ins.operation == OT.VEC_ADD
    assert pipe.stages[-2].ins.state == IS.DONE
    assert np.all(core.get_reg("vrA").data == base[:length])

    # check register before instruction reaches writeback
    while pipe.stages[-1].ins is None:
        assert np.all(core.get_reg("vrA").data == base[:length])
        core.tick()
        mem.tick()

    assert pipe.stages[-1].ins is not None
    assert pipe.stages[-1].ins.operation == OT.VEC_ADD
    assert pipe.stages[-1].ins.state == IS.DONE
    assert np.all(core.get_reg("vrA").data == base[:length])

    core.tick()
    mem.tick()

    assert np.all(
        core.get_reg("vrA").data == base[:length] + base[length : 2 * length]
    )
