class Location:
    def __init__(self, channel: int, rank: int, bankgroup: int, bank: int):
        self.channel = channel
        self.rank = rank
        self.bankgroup = bankgroup
        self.bank = bank


    def __getitem__(self, i: int) -> int:
        match i:
            case 0:
                return self.channel
            case 1:
                return self.rank
            case 2:
                return self.bankgroup
            case 3:
                return self.bank
            case _:
                return 0
