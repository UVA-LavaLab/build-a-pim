class PimRegError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class PimCmdNotSupportedError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class PimCmdNotImplementedError(Exception):
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
