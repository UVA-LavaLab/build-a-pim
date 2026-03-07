from lib.memsys import MemSystem

class Core:
    def __init__(self, location: tuple[int, int, int, int]):
        self.channel = location[0]
        self.rank = location[1]
        self.bankgroup = location[2]
        self.bank = location[3]

        self.gdl = []
        self.instruction_queue = []

    def tick(self, mem: MemSystem):
        return
