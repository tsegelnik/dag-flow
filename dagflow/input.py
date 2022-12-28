from typing import Iterator, Tuple, Optional, Union
from numpy import zeros
from numpy.typing import DTypeLike, NDArray

from .edges import EdgeContainer
from .exception import (
    ClosedGraphError,
    ReconnectionError,
    AllocationError,
    ConnectionError,
    InitializationError,
)
from .output import Output
from .shift import lshift
from .tools import StopNesting
from .types import InputT, NodeT


class Input:
    _own_data: Optional[NDArray] = None
    _own_dtype: Optional[DTypeLike] = None
    _own_shape: Optional[Tuple[int, ...]] = None

    _node: Optional[NodeT]
    _name: Optional[str]

    _parent_output: Optional[Output]
    _child_output: Optional[Output]

    _allocatable: bool = False
    _owns_data: bool = False

    _debug: bool = False

    def __init__(
        self,
        name: Optional[str] = None,
        node: Optional[NodeT] = None,
        *,
        child_output: Optional[Output] = None,
        parent_output: Optional[Output] = None,
        debug: Optional[bool] = None,
        allocatable: bool = False,
        data: Optional[NDArray] = None,
        dtype: Optional[DTypeLike] = None,
        shape: Optional[Tuple[int, ...]] = None,
    ):
        if data is not None and (
            allocatable or dtype is not None or shape is not None
        ):
            raise InitializationError(input=input, node=node)

        self._name = name
        self._node = node
        self._child_output = child_output
        self._parent_output = parent_output
        self._allocatable = allocatable
        self._debug = (
            debug if debug is not None else node.debug if node else False
        )

        self._own_dtype = dtype
        self._own_shape = shape
        if data is not None:
            self.set_own_data(data, owns_data=True)

    def __str__(self) -> str:
        return (
            f"→○ {self._name}"
            if self._owns_data is None
            else f"→● {self._name}"
        )

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def own_data(self):
        return self._own_data

    @property
    def owns_data(self) -> bool:
        return self._owns_data

    def set_own_data(self, data, *, owns_data: bool):
        if self.closed:
            raise ClosedGraphError(
                "Unable to set input data.", node=self._node, input=self
            )
        if self._own_data is not None:
            raise AllocationError(
                "Input already has data.", node=self._node, input=self
            )

        self._own_data = data
        self._owns_data = owns_data
        self._own_dtype = data.dtype
        self._own_shape = data.shape

    @property
    def closed(self):
        return self._node.closed if self.node else False

    @property
    def own_dtype(self):
        return self._own_dtype

    @property
    def own_shape(self):
        return self._own_shape

    def set_child_output(
        self, child_output: Output, force: bool = False
    ) -> None:
        if not self.closed:
            return self._set_child_output(child_output, force)
        raise ClosedGraphError(input=self, node=self.node, output=child_output)

    def _set_child_output(
        self, child_output: Output, force: bool = False
    ) -> None:
        if self.child_output and not force:
            raise ReconnectionError(output=self.child_output, node=self.node)
        self._child_output = child_output
        child_output.parent_input = self

    def set_parent_output(
        self, parent_output: Output, force: bool = False
    ) -> None:
        if not self.closed:
            return self._set_parent_output(parent_output, force)
        raise ClosedGraphError(
            input=self, node=self.node, output=parent_output
        )

    def _set_parent_output(
        self, parent_output: Output, force: bool = False
    ) -> None:
        if self.connected() and not force:
            raise ReconnectionError(output=self._parent_output, node=self.node)
        self._parent_output = parent_output

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name) -> None:
        self._name = name

    @property
    def node(self) -> NodeT:
        return self._node

    @property
    def parent_node(self) -> NodeT:
        return self._parent_output.node

    @property
    def logger(self):
        return self._node.logger

    @property
    def child_output(self) -> InputT:
        return self._child_output

    @property
    def invalid(self) -> bool:
        """Checks validity of the parent output data"""
        return self._parent_output.invalid

    @property
    def has_data(self) -> bool:
        return self._own_data is not None

    @property
    def allocatable(self) -> bool:
        return self._allocatable

    @property
    def debug(self) -> bool:
        return self._debug

    @invalid.setter
    def invalid(self, invalid) -> None:
        """Sets the validity of the current node"""
        self._node.invalid = invalid

    @property
    def parent_output(self) -> Output:
        return self._parent_output

    @property
    def data(self):
        return self._parent_output.get_data_unsafe()

    @property
    def dtype(self):
        return self._parent_output.dtype

    @property
    def shape(self):
        return self._parent_output.shape

    @property
    def tainted(self) -> bool:
        return self._parent_output.tainted

    def touch(self):
        return self._parent_output.touch()

    def taint(self, force: bool = False) -> None:
        self._node.taint(force)

    def taint_type(self, force: bool = False) -> None:
        self._node.taint_type(force)

    def connected(self) -> bool:
        return bool(self._parent_output)

    def deep_iter_inputs(self, disconnected_only=False):
        if disconnected_only and self.connected():
            return iter(tuple())
        raise StopNesting(self)

    def deep_iter_child_outputs(self):
        if self._child_output:
            raise StopNesting(self._child_output)
        return iter(tuple())

    def __lshift__(self, other):
        """
        self << other
        """
        return lshift(self, other)

    def __rrshift__(self, other):
        """
        other >> self
        """
        return lshift(self, other)

    def allocate(self, **kwargs) -> bool:
        if not self._allocatable or self.has_data:
            return True

        if self._own_shape is None or self._own_dtype is None:
            raise AllocationError(
                "No shape/type information provided for the Input",
                node=self._node,
                output=self,
            )
        try:
            self._own_data = zeros(self._own_shape, self._own_dtype, **kwargs)
        except Exception as exc:
            raise AllocationError(
                f"Input: {exc.args[0]}", node=self._node, input=self
            ) from exc

        return True


class Inputs(EdgeContainer):
    _dtype = Input

    def __init__(self, iterable=None):
        super().__init__(iterable)

    def __str__(self):
        return f"→[{tuple(obj.name for obj in self)}]○"

    def deep_iter_inputs(
        self, disconnected_only: bool = False
    ) -> Iterator[Input]:
        for input in self:
            if disconnected_only and input.connected():
                continue
            yield input

    def deep_iter_child_outputs(self) -> Iterator[Union[Input, Output]]:
        for child_output in self:
            yield child_output.child_output

    def touch(self) -> None:
        for input in self:
            input.touch()
