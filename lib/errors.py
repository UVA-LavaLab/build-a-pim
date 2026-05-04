class PimRegError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class PimCmdNotSupportedError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class PimCmdNotImplementedError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class PimAccessOutOfBoundsError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class PimCmdMalformedError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class MemCmdMalformedError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class PimInstructionUnsupportedError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class PimInstructionMalformedError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class PipelineExitCallbackNotDefinedError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class MisalignedMemWriteError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class PimInvalidRegisterIDError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class PimCrammedResponseError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class AddressMappingNotAscendingError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class AllocationStrategyNotSupportedError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class PimMmapOutOfBoundsError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
