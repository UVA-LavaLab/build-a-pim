import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.dramsim import callback_t
from lib.memsys import MemSystem
from lib.monad import DataStatus, DataWrapper, DataSetter, Ptr, DataStructureContainer
from lib.cores.mode_switcher import Core as MS
from lib.cores.components.scratchpad import Scratchpad
from lib.cores.instructions import Instruction, OpType
from lib.types import Location
from lib.controller.commands import Command, CommandType
from lib.device.base import BaseDevice
import numpy as np
import math
import random

TFAW = 4

def benchmark():
    # this is lazy, but we use the device object to create cores and map to memory
    dev = BaseDevice(MS, "./dramsim3/configs/HBM2_8Gb_x128.ini")
    p_mem: Ptr[MemSystem] = Ptr(dev.mem)
    cores: list[MS] = dev.cores

    test_list = np.arange(65536 * 2)
    id = dev.instant_place_data(test_list)
    max_gdl_chunks_per_core = (
        p_mem().address_mapper.boundaries[1] - p_mem().address_mapper.boundaries[0] - 1
    )

    for core in cores:
        # open a random row on each core (ignore tFAW and other timings)
        core.add_instruction(
            OpType.READ,
            addr=random.randint(0, max_gdl_chunks_per_core),
        )

    # tick the device forward to start the above reads
    for _ in range(3):
        dev.tick()

    # now we tick all cores forward until the read transactions are complete
    while not dev.all_cores_idle():
        dev.tick()

    # tick some more to ensure fair comparison
    for _ in range(50):
        dev.tick()

    start_time: int = dev.cycle
    for core in cores:
        # no response can be returned here, we're just starting to switch to
        # mem mode
        _ = core.tick(cmd=Command(CommandType.SWITCH_MODE_MEM))

    # make a random list of 128 mem requests to serve this is a middleground
    # for the number of requests we would likely be able to buffer
    mem_rq: list[tuple[int, int]] = [
        (
            random.randint(0, len(cores) - 1),
            random.randint(0, max_gdl_chunks_per_core - 1),
        )
        for _ in range(128)
    ]
    req_in: int = 0
    req_out: int = 0
    first_mem_req_serviced_after: int = -1
    tfaw_cooldown: int = 0

    gang_priorities: list[int] = list(range(int(len(cores) / 4)))
    gang_counts: list[int] = [0 for _ in cores]
    for v in mem_rq:
        core = v[0]
        gang_counts[int(core / 4)] += 1

    # this sorts by fewest number of pending reads
    gang_priorities = [x for _, x in sorted(zip(gang_counts, gang_priorities))]

    # this lets us properly format the emitted command
    def get_loc_from_gang(gang: int):
        return cores[gang * 4].location

    ping_in: int = 0
    ping_out: int = 0

    # now the controller knows that every core is in dirty mode
    dirty_bits: list[bool] = [False for _ in cores]
    gangs_covered: set[int] = set()
    mode_switch_complete: int = -1
    # this loop implements the controller policy surrounding mode switching
    # because all other parameters (intra-bank timings) are checked by
    # dramsim3, we can totally ignore them and only check tFAW

    # please also note that this level of latency can be achieved *without*
    # pinging any of the banks
    while req_out < len(mem_rq):
        cmd = None
        if tfaw_cooldown <= 0 and req_in < len(mem_rq):
            # if we aren't either waiting for a request or on cooldown, we
            # should send another one. these requests will automatically gang
            # as per AiM's convention
            cmd = Command(
                CommandType.MEM_READ,
                operand_1=mem_rq[req_in][1],
                location=cores[mem_rq[req_in][0]].location,
            )
            # helps avoid pinging and activating the same banks
            gangs_covered.add(int(mem_rq[req_in][0] / 4))
            tfaw_cooldown = TFAW
            req_in += 1
        if ping_out == ping_in and cmd is None:
            cmd = Command(
                CommandType.PIM_BANK_PING,
                location=get_loc_from_gang(gang_priorities[ping_in]),
            )
            while gang_priorities[ping_in] in gangs_covered and ping_out < len(
                gang_priorities
            ):
                ping_in += 1
                ping_out += 1
            if ping_in < len(gang_priorities):
                gangs_covered.add(gang_priorities[ping_in])
                ping_in += 1

        bits_in_cycle: int = 0
        for i, core in enumerate(cores):
            r = core.tick(cmd)
            if r is not None:
                bits_in_cycle += r.bits
                if len(r.bytes) != 0:
                    # mem case
                    # if this is the first memory request serviced, record the time
                    first_mem_req_serviced_after = (
                        first_mem_req_serviced_after
                        if first_mem_req_serviced_after != -1
                        else core.cycle - start_time
                    )
                    req_out += 1
                    # set all of the dirty bits for the corresponding gang to
                    # True
                    floor_i = int(i / 4) * 4
                    for j in range(floor_i, floor_i + 4):
                        dirty_bits[j] = True
                else:
                    # ping case
                    ping_out += 1
                    # set all of the dirty bits for the corresponding gang to
                    # True. this will do redundant work on the simluator's end,
                    # but it is relatively minimal and can be optimized out
                    # later
                    floor_i = int(i / 4) * 4
                    for j in range(floor_i, floor_i + 4):
                        dirty_bits[j] = True

        p_mem().tick()
        # on the first cycle where this is true, we have technically completed
        # the mode switch
        if all(dirty_bits) and mode_switch_complete == -1:
            mode_switch_complete = p_mem().m_cycle - start_time

        # on every device tick, we subtract one from the tfaw cooldown
        tfaw_cooldown -= 1

    print("First request served after:", first_mem_req_serviced_after)
    print("Mode switch completed at:", mode_switch_complete)
    print("Cycles taken:", p_mem().m_cycle)
    print("Pings issued:", ping_out)
    print("Requests served:", req_out)
    return (first_mem_req_serviced_after, mode_switch_complete)


if __name__ == "__main__":
    firsts: list[int] = []
    finishes: list[int] = []
    for i in range(100):
        first, done = benchmark()
        firsts.append(first)
        finishes.append(done)

    cleaned_finishes = [f for f in finishes if f != -1]
    partial = len([f for f in finishes if f == -1])

    print("------------first response time------------")
    print(f"Average: {sum(firsts) / len(firsts)}")
    print(f"Min: {min(firsts)}")
    print(f"Max: {max(firsts)}")
    print("------------mode switch finish time------------")
    print(f"Average: {sum(cleaned_finishes) / len(cleaned_finishes)}")
    print(f"Min: {min(cleaned_finishes)}")
    print(f"Max: {max(cleaned_finishes)}")
    print(f"Percent partial: {partial}")
