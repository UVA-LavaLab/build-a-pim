from lib.errors import PimRegError


class Location:
    def __init__(self, channel: int, rank: int, bankgroup: int, bank: int):
        self.channel: int = channel
        self.rank: int = rank
        self.bankgroup: int = bankgroup
        self.bank: int = bank

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
                return -1


class PimRegType[PimCore]:
    def __init__(self, ident: str):
        if hasattr(PimCore, ident):
            self.ident: str = ident
        else:
            raise PimRegError(
                f"PimCore {PimCore.__name__} does not have register with identifier {ident}"
            )
