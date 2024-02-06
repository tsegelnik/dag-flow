from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from .input import Input
    from .node import Node
    from .output import Output


class DagflowError(RuntimeError):
    node: Optional[Node]
    input: Optional[Input]
    output: Optional[Output]

    def __init__(
        self,
        message: str,
        node: Optional[Node] = None,
        *,
        input: Optional[Input] = None,
        output: Optional[Output] = None,
        args: Optional[Tuple[str, ...]] = None,
    ):
        if node:
            message = f"{message} [node={getattr(node, 'name', node)}]"
        if input:
            message = f"{message} [input={getattr(input, 'name', input)}]"
        if output:
            message = f"{message} [output={getattr(output, 'name', output)}]"
        super().__init__(message)
        self.node = node
        self.input = input
        self.output = output

        if node is not None and hasattr(node, "_exception"):
            if args is not None:
                node._exception = "\\n".join((message,) + args)
            else:
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
    def __init__(self, message: Optional[str] = None, *args, **kwargs):
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
