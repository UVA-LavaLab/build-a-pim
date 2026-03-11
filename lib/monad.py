from typing import Any, Callable
from enum import Enum


class DataStatus(Enum):
    COLD = 0
    READY = 1


class DataStructureContainer:
    def __init__(self, data_structure: Any, element_size_bytes: int):
        self.data_structure: Any = data_structure
        self.element_size_bytes: Any = element_size_bytes

    def __getitem__(self, key: int):
        return self.data_structure[key]

    def __str__(self):
        return str(self.data_structure)


class DataSetter:
    def __init__(self, in_wrapper: DataWrapper):
        self.input: DataWrapper = in_wrapper
        self.output: DataWrapper = DataWrapper([])

class DataWrapper:
    def __init__(self, data: Any, update_func:Callable[[], bool] | None = None):
        self.data = data
        self.status = DataStatus.COLD
        if update_func is not None:
            self.update_func: Callable[[], bool] = update_func
        else:
            def u():
                return True
            self.update_func = u

    # forward the [] operator to the contained value
    def __getitem__(self, key: int) -> Any:
        if self.status != DataStatus.READY:
            raise Exception(
                "Failed to access data it index, data not ready. Index was:", key
            )
        return self.data[key]

    def __str__(self) -> str:
        if self.status == DataStatus.COLD:
            stat = "COLD"
        else:
            stat = "READY"
        return f"{str(self.data)} -> status={stat}"

    def update_status(self):
        if self.update_func():
            self.status = DataStatus.READY

    @property
    def is_cold(self):
        self.update_status()
        return self.status == DataStatus.COLD

    @property
    def is_ready(self) -> bool:
        self.update_status()
        return self.status == DataStatus.READY

    def set_ready(self):
        self.status = DataStatus.READY

    def set_cold(self):
        self.status = DataStatus.COLD

    def set_warm(self):
        self.status = DataStatus.WARM

    def raise_level(self):
        match self.status:
            case DataStatus.COLD:
                self.set_ready()
            case _:
                pass

    def lower_level(self):
        match self.status:
            case DataStatus.READY:
                self.set_cold()
            case _:
                pass
