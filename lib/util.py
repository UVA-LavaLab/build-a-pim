from typing import Any

def rev_enum(data: list[Any]):
    for i in range(len(data) - 1, -1, -1):
        yield (i, data[i])
