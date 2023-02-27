from typing import Optional
from .types import NodeT, InputT, OutputT


class DagflowError(Exception):
    node: Optional[NodeT]
    input: Optional[InputT]
    output: Optional[OutputT]

    def __init__(
        self,
        message: str,
        node: Optional[NodeT] = None,
        *,
        input: Optional[InputT] = None,
        output: Optional[OutputT] = None,
    ):
        if node:
            message = f"{message} [node={node.name if 'name' in dir(node) else node}]"
        if input:
            message = f"{message} [input={input.name if 'name' in dir(input) else input}]"
        if output:
            message = f"{message} [output={output.name if 'name' in dir(output) else output}]"
        super().__init__(message)
        self.node = node
        self.input = input
        self.output = output

        if node is not None:
            node._exception = message

class CriticalError(DagflowError):
    pass


class NoncriticalError(DagflowError):
    pass

class InitializationError(CriticalError):
    def __init__(self, message: Optional[str] = None, *args, **kwargs):
        if not message:
            message = "Wrong initialization!"
        super().__init__(message, *args, **kwargs)


class AllocationError(CriticalError):
    def __init__(self, message: Optional[str] = None, *args, **kwargs):
        if not message:
            message = "Unable to allocate memory!"
        super().__init__(message, *args, **kwargs)

class ClosingError(CriticalError):
    def __init__(self, message: Optional[str] = None, *args, **kwargs):
        if not message:
            message = "An exception occured during closing procedure!"
        super().__init__(message, *args, **kwargs)

class OpeningError(CriticalError):
    def __init__(self, message: Optional[str] = None, *args, **kwargs):
        if not message:
            message = "An exception occured during opening procedure!"
        super().__init__(message, *args, **kwargs)

class ClosedGraphError(CriticalError):
    def __init__(self, message: Optional[str] = None, *args, **kwargs):
        if not message:
            message = "Unable to modify a closed graph!"
        super().__init__(message, *args, **kwargs)

class UnclosedGraphError(CriticalError):
    def __init__(self, message : Optional[str]=None, *args, **kwargs):
        if not message:
            message = "The graph is not closed!"
        super().__init__(message, *args, **kwargs)


class TypeFunctionError(CriticalError):
    def __init__(self, message: Optional[str] = None, *args, **kwargs):
        if not message:
            message = "An exception occurred during type function processing!"
        super().__init__(message, *args, **kwargs)

class ReconnectionError(CriticalError):
    def __init__(self, message: Optional[str] = None, *args, **kwargs):
        if not message:
            message = "The object is already connected!"
        super().__init__(message, *args, **kwargs)

class ConnectionError(CriticalError):
    def __init__(self, message: Optional[str] = None, *args, **kwargs):
        if not message:
            message = "An exception occurred during connection!"
        super().__init__(message, *args, **kwargs)
