from typing import Callable, Generic, TypeVar, override, Literal, Any
from enum import Enum
import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray
from numpy import generic

T = TypeVar("T")


class Ptr(Generic[T]):
    """
    A wrapper class which allows for sharing of instances between classes.

    Instantiation example:
    p: Ptr[ClassType] = Ptr(cls)

    Dereferencing example:
    cls = p.deref()
    cls = p()
    cls = (*p,)[0]
    cls = [*p][0]
    """

    def __init__(self, obj: T):
        self._internal: T = obj

    def deref(self) -> T:
        return self._internal

    def __call__(self) -> T:
        return self._internal

    def __iter__(self):
        yield self._internal

    def __class_getitem__(cls, types: type) -> type:
        if isinstance(types, tuple):
            raise TypeError(f"Ptr only accepts a single type parameter.")
        return type(f"Ptr[{types.__name__}]", (cls,), {"_types": (types,)})


class DataStatus(Enum):
    COLD = 0
    READY = 1


class DataStructureContainer:
    def __init__(self, data_structure: NDArray[generic], endianness: Literal[">", "<"] = "<"):
        self.data_structure: NDArray[generic] = data_structure
        self.endianness: Literal[">", "<"] = endianness

    @property
    def element_size_bytes(self) -> int:
        return self.data_structure.dtype.itemsize

    def __getitem__(self, key: int | tuple[int, npt.DTypeLike]) -> generic:
        if isinstance(key, int):
            dt = np.dtype(np.int32)
        else:
            dt = np.dtype(key[1])
            key: int = key[0]
        return np.frombuffer(self.data_structure, dtype=dt.newbyteorder())[key]

    @override
    def __str__(self):
        return str(self.data_structure)





class DataWrapper:
    def __init__(
        self,
        data: NDArray[generic] | list[Any],
        update_func: Callable[[], bool] | None = None,
        endianness: Literal[">", "<"] = "<",
    ):
        if isinstance(data, list):
            data = np.array(data, dtype=np.int32)
        self.data: memoryview = data.data
        self.status: DataStatus = DataStatus.COLD
        if update_func is not None:
            self.update_func: Callable[[], bool] = update_func
        else:

            def u():
                return False

            self.update_func = u
        self.endianness: Literal[">", "<"] = endianness

    # forward the [] operator to the contained value
    def __getitem__(self, key: int | tuple[int, npt.DTypeLike]) -> generic:
        if self.status != DataStatus.READY:
            raise Exception(
                "Failed to access data in index, data not ready. Index was:", key
            )
        if isinstance(key, int):
            dt = np.dtype(np.int32)
        else:
            dt = np.dtype(key[1])
            key: int = key[0]
        data = np.frombuffer(self.data, dtype=dt.newbyteorder(self.endianness))
        return data[key]

    def __setitem__(
        self, key: int | tuple[int, npt.DTypeLike], value: Any
    ) -> None:
        if isinstance(key, int):
            dt = np.dtype(np.int32)
        else:
            dt = np.dtype(key[1])
            key: int = key[0]
        data = np.frombuffer(self.data, dtype=dt.newbyteorder(self.endianness))
        data[key] = value

    @override
    def __str__(self) -> str:
        return self.str_as_type(np.uint8)

    def str_as_type(self, dtype: npt.DTypeLike) -> str:
        if self.status == DataStatus.COLD:
            stat = "COLD"
        else:
            stat = "READY"
        return f"{str(np.frombuffer(self.data, dtype=dtype))} -> status={stat}"

    def update_status(self):
        if self.update_func():
            self.status = DataStatus.READY

    @property
    def is_cold(self):
        self.update_status()
        return self.status == DataStatus.COLD

    def is_ready(self) -> bool:
        self.update_status()
        return self.status == DataStatus.READY

    def set_ready(self):
        self.status = DataStatus.READY

    def set_cold(self):
        self.status = DataStatus.COLD

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

class DataSetter:
    def __init__(self, in_wrapper: DataWrapper):
        self.input: DataWrapper = in_wrapper
        self.output: DataWrapper = DataWrapper(np.array([]))
