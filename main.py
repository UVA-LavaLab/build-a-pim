from lib.dramsim import callback_t
from lib.memsys import MemSystem
from lib.monad import DataStatus, DataWrapper, DataSetter, Ptr
from lib.cores.lobsta import Core
from lib.cores.instructions import Instruction, OpType
from lib.types import Location
from lib.controller.commands import Command, CommandType

if __name__ == "__main__":
    mem = MemSystem("./dramsim3/configs/DDR4_8Gb_x16_3200.ini", ".", nd_log=True)
    mem.toggle_pim_mode()
    test_list = list(range(32))
    mem.add_data_structure(test_list, 4)
    mem.mmap(0, 0, 0, 0, 0, data_index=0, length=len(test_list) * 4, offset=0)

    core = Core((0, 0, 0, 0), Ptr(mem))
    core.add_instruction(OpType.NOP)
    core.add_instruction(OpType.READ, operands=[0x0, "regA"])
    core.add_instruction(OpType.READ, operands=[0x10])
    core.add_instruction(OpType.ADD, operands=["regA", 0x10])

    i = 0
    while len(core.instruction_queue) > 0 or not core.pipeline.is_empty():
        core.tick(Command(CommandType.PIM_ADD))
        mem.tick()
        print(core.pipeline)
        print([str(i) for i in core.instruction_queue])
        print(core.cycle)
        print(core.gdl)
        print(mem.nd_log)
        print("core's regA", core.regA)
        print("----------------")
        i += 1
        if i == 80:
            break

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
