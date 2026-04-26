from lib.cores.components.base import BaseCore
from lib.memsys import MemSystem
from lib.monad import Ptr
from lib.controller.controller import Controller, ControllerState, BaselineState, Transaction
from lib.controller.response import Response
from lib.errors import PimCrammedResponseError


def crammed[T: BaseCore](dev: BaseDevice[T], bits: int):
    raise PimCrammedResponseError(
        f"Cumulative bits sent along the GDL {bits} is larger than device GDL"
        + f" width {dev.mem.m_gdl_width} during cycle {dev.cycle}"
    )

class BaseDevice[T: BaseCore]:
    def __init__(self, core_type: type[T], config: str, cores: list[T] | None = None):
        # TODO: incorporate logic for different-frequency components
        self.cycle: int = 0
        self.mem: MemSystem = MemSystem(config, ".")
        p_mem = Ptr(self.mem)

        # for clarity
        n_bank = p_mem().c_num_banks_per_group
        n_bankgroup = p_mem().c_num_bankgroups_per_rank
        n_rank = p_mem().c_num_ranks
        n_channel = p_mem().c_num_channels

        # create a list of cores if none are provided (useful for different core mappings)
        self.cores: list[T] = (
            [
                core_type((c, r, bg, b), p_mem)
                for c in range(n_channel)
                for r in range(n_rank)
                for bg in range(n_bankgroup)
                for b in range(n_bank)
            ]
            if cores is None
            else cores
        )

        self.controller: Controller[BaselineState] = Controller(
            ControllerState(
                BaselineState(pim_mode=p_mem().get_pim_mode(), threshold=2)
            ),
            command_functions=[],
            mem_pointer=p_mem,
        )

    def tick(self, trans: Transaction | None = None):
        bits: int = 0
        cmd = self.controller.tick(trans)

        responses: list[Response] = []
        for core in self.cores:
            r = core.tick(cmd)
            if r is not None:
                responses.append(r)
                bits += r.bits

        if bits > self.mem.m_gdl_width:
            crammed(self, bits)

        for r in responses:
            self.controller.push_response(r)
        self.mem.tick()
        self.cycle += 1
