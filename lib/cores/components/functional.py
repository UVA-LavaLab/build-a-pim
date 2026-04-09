from lib.errors import PimInstructionUnsupportedError
from lib.cores.instructions import Instruction
from lib.cores.components.base import BaseCore

# TODO: determine how to do bounds checking for addresses passed here,
# currently just assumes that the requested address is in the GDL already
# ALSO: add type checking before performing operations
def evaluate_instruction(core: BaseCore, f, ins: Instruction):
    """
    A higher-order function which accepts a core, a function (f), and an instruction,
    then executes that function based on the operands passed via the instruction.
    """
    assert len(ins.operands) >= 2
    if isinstance(ins.operands[0], str) and isinstance(
        ins.operands[1], int
    ):
        dst = ins.op_vals[ins.operands[0]]
        for i in range(len(ins.op_vals[ins.operands[1]].data)):
            dst[i] = f(dst[i], ins.op_vals[ins.operands[1]][i])
        core.set_reg("gdl", dst)
    elif isinstance(ins.operands[0], int) and isinstance(
        ins.operands[1], str
    ):
        reg = ins.op_vals[ins.operands[1]]
        for i in range(len(ins.op_vals[ins.operands[0]].data)):
            ins.op_vals[ins.operands[0]][i] = f(
                ins.op_vals[ins.operands[0]][i], reg[i]
            )
        core.set_reg("gdl", reg)
    elif isinstance(ins.operands[0], str) and isinstance(
        ins.operands[1], str
    ):
        reg0 = ins.fetch_operands(0)
        reg1 = ins.fetch_operands(1)
        for i in range(len(reg0.data)):
            reg0[i] = f(reg0[i], reg1[i])
        core.set_reg(ins.operands[0], reg0)
    else:
        raise PimInstructionUnsupportedError(
            "Arithmetic operations between two memory locations unsupported."
        )

