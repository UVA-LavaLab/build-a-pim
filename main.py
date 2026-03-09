from lib.dramsim import callback_t
from lib.memsys import MemSystem
from lib.monad import DataStatus
from lib.cores.lobsta import Core
from lib.cores.instructions import Instruction, OpType

if __name__ == "__main__":
    mem = MemSystem("./dramsim3/configs/DDR4_8Gb_x16_3200.ini", ".", nd_log=True)
    test_list = list(range(32))
    mem.add_data_structure(test_list, 4)
    mem.mmap(0, 0, 0, 0, 0, data_index=0, length=len(test_list) * 4, offset=0)

    core = Core((0, 0, 0, 0))
    core.add_instruction(OpType.NOP)
    core.add_instruction(OpType.READ, operands=[0x0])
    core.tick(mem)
    core.tick(mem)
    # a = mem.bank_local_addr(0, 0, 0, 3, 4)
    # print(mem.loc_from_addr(a))
    #
    # val = mem[0, 0, 0, 1, 0x10]
    # val2 = None
    # print(val)
    # mem.toggle_pim_mode()
    # for _ in range(2):
    #     mem.tick(until_event=True)
    #     print(mem.nd_log[0][0][0][1])
    #     if val.is_ready:
    #         print(val, mem.m_cycle)
    #         val2 = mem[0, 0, 0, 1, 0x0]
    #     if val2 is not None and val2.is_ready:
    #         print(val.is_ready)
    #         print(val, val2, mem.m_cycle)
