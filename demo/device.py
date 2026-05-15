import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.memsys import MemSystem
from lib.containers import Ptr
from lib.cores.ins_stream_bank_simd import Core
from lib.controller.commands import Command, CommandType
from lib.device.device_commands import DeviceCommand
from lib.address.allocation import pim_device_place_data
from lib.device.base import BaseDevice as Device
import numpy as np
import numpy.typing as npt
import random

if __name__ == "__main__":
    # initialize the device
    dev = Device(Core, "./dramsim3/configs/HBM2_8Gb_x128.ini")

    # generate input vectors (size is NOT an even multiple of GDL width * num
    # cores, so one core will receive one less GDL-width chunk) all data is on
    # host to start
    vec_len = 65536 - 16
    in1: npt.NDArray[np.int32] = np.random.randint(0, 5, size=vec_len, dtype=np.int32)
    in2: npt.NDArray[np.int32] = np.random.randint(0, 5, size=vec_len, dtype=np.int32)
    dst: npt.NDArray[np.int32] = np.zeros(vec_len, dtype=np.int32)

    # copy the data to the device (untimed API)
    i1id = dev.instant_place_data(in1)
    i2id = dev.instant_place_data(in2)
    dstid = dev.instant_place_data(dst)

    # build a transaction (tells the device which kernel to run and on which object)
    d = DeviceCommand(
        CommandType.PIM_ADD, op1_id=i1id, op2_id=i2id, dst_id=dstid
    )
    # add the transaction to the device
    assert dev.add_transaction(d)

    for _ in range(10):
        dev.tick()

    while not dev.all_cores_idle() or not dev.transaction_queue_empty():
        dev.tick()

    # let's look at what we did!
    print("vector length:", vec_len)
    print("--------------------------------")
    print("input 1:", in1)
    print("input 2:", in2)
    print("device result:", dev.mem.stored_data_structures[dstid].data_structure)
    print("--------------------------------")
    print(
        "Output matches expected:",
        np.all(
            dev.mem.stored_data_structures[dstid].data_structure
            == in1 + in2
        ),
    )
    print("Cycles taken (also ns in this case):", dev.cycle, "(cycles & ns)")
