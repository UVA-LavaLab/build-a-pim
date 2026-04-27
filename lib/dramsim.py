import ctypes
import platform
from typing import TypeAlias, Callable

dramsim3 = ctypes.cdll.LoadLibrary(
    "./dramsim3/libdramsim3.dylib"
    if platform.system() == "Darwin"
    else "./dramsim3/libdramsim3.so"
)
callback_t = ctypes.CFUNCTYPE(None, ctypes.c_uint64)
CallbackType: TypeAlias = Callable[[int], None]

# define argtypes
dramsim3.memsys_create.argtypes = [
    ctypes.c_char_p,
    ctypes.c_char_p,
    callback_t,
    callback_t,
]
dramsim3.memsys_print_stats.argtypes = [ctypes.c_void_p]
dramsim3.memsys_destroy.argtypes = [ctypes.c_void_p]
dramsim3.memsys_tick.argtypes = [ctypes.c_void_p]
dramsim3.memsys_register_callbacks.argtypes = [ctypes.c_void_p, callback_t, callback_t]
dramsim3.memsys_add_transaction.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_bool,
    ctypes.c_bool,
]
dramsim3.memsys_add_transaction_to_bank.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_bool,
    ctypes.c_bool,
]
dramsim3.memsys_toggle_mode.argtypes = [ctypes.c_void_p]
dramsim3.memsys_set_pim_mode.argtypes = [ctypes.c_void_p, ctypes.c_bool]
dramsim3.memsys_get_pim_mode.argtypes = [ctypes.c_void_p]
dramsim3.memsys_get_cycle.argtypes = [ctypes.c_void_p]
dramsim3.memsys_get_ranks.argtypes = [ctypes.c_void_p]
dramsim3.memsys_get_channels.argtypes = [ctypes.c_void_p]
dramsim3.memsys_get_banks_per_bankgroup.argtypes = [ctypes.c_void_p]
dramsim3.memsys_get_bankgroups_per_rank.argtypes = [ctypes.c_void_p]
dramsim3.memsys_get_address_from_physical_location.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
]
dramsim3.memsys_get_byte_range_from_bank.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_int64),
    ctypes.POINTER(ctypes.c_size_t),
]
dramsim3.memsys_mmap.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_int64,
    ctypes.c_size_t,
    ctypes.c_size_t,
]
dramsim3.memsys_munmap.argtypes = [
    ctypes.c_void_p,  # memsys
    ctypes.c_uint64,  # channel
    ctypes.c_uint64,  # rank
    ctypes.c_uint64,  # bankgroup
    ctypes.c_uint64,  # bank
    ctypes.c_size_t,  # base address
    ctypes.c_size_t,  # length
]
dramsim3.memsys_get_config_property.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
dramsim3.memsys_get_physical_location_from_address.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int64),
    ctypes.POINTER(ctypes.c_int64),
    ctypes.POINTER(ctypes.c_int64),
    ctypes.POINTER(ctypes.c_int64),
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_uint64,
]
dramsim3.memsys_get_canonical_from_phys.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
]
dramsim3.memsys_get_canonical_from_global.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint64,
]
dramsim3.memsys_get_tck.argtypes = [
    ctypes.c_void_p,
]
dramsim3.memsys_get_active_row.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_uint64,
]

# define restypes
dramsim3.memsys_create.restype = ctypes.c_void_p
dramsim3.memsys_add_transaction.restype = ctypes.c_bool
dramsim3.memsys_add_transaction_to_bank.restype = ctypes.c_bool
dramsim3.memsys_get_pim_mode.restype = ctypes.c_bool
dramsim3.memsys_get_cycle.restype = ctypes.c_uint64

dramsim3.memsys_get_ranks.restype = ctypes.c_uint64
dramsim3.memsys_get_channels.restype = ctypes.c_uint64
dramsim3.memsys_get_banks_per_bankgroup.restype = ctypes.c_uint64
dramsim3.memsys_get_bankgroups_per_rank.restype = ctypes.c_uint64
dramsim3.memsys_get_address_from_physical_location.restype = ctypes.c_uint64
dramsim3.memsys_get_config_property.restype = ctypes.c_int
dramsim3.memsys_get_tck.restype = ctypes.c_float
dramsim3.memsys_get_active_row.restype = ctypes.c_int
dramsim3.memsys_get_canonical_from_phys.restype = ctypes.c_uint64
dramsim3.memsys_get_canonical_from_global.restype = ctypes.c_uint64
