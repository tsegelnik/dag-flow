from typing import Iterator, Optional, Tuple, Union
from numpy import zeros
from numpy.typing import DTypeLike, NDArray

from dagflow.datadescriptor import DataDescriptor

from .edges import EdgeContainer
from .exception import (
    ClosedGraphError,
    ReconnectionError,
    AllocationError,
    InitializationError,
)
from .output import Output
from .shift import lshift
from .iter import StopNesting
from .types import EdgesLike, InputT, NodeT, ShapeLike


class Input:
    _own_data: Optional[NDArray] = None
    _own_dd: DataDescriptor

    _node: Optional[NodeT]
    _name: Optional[str]

    _parent_output: Optional[Output]
    _child_output: Optional[Output]

    _allocatable: bool = False
    _owns_buffer: bool = False

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
        dtype: DTypeLike = None,
        shape: Optional[ShapeLike] = None,
        axes_edges: Optional[Tuple[EdgesLike]] = None,
        axes_nodes: Optional[Tuple[EdgesLike]] = None,
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
        if debug is not None:
            self._debug = debug
        elif node:
            self._debug = node.debug
        else:
            self._debug = False

        self._own_dd = DataDescriptor(dtype, shape, axes_edges, axes_nodes)

        if data is not None:
            self.set_own_data(data, owns_buffer=True)

    def __str__(self) -> str:
        return (
            f"→○ {self._name}"
            if self._owns_buffer is None
            else f"→● {self._name}"
        )

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def own_data(self) -> Optional[NDArray]:
        return self._own_data

    @property
    def own_dd(self) -> DataDescriptor:
        return self._own_dd

    @property
    def owns_buffer(self) -> bool:
        return self._owns_buffer

    def set_own_data(
        self,
        data,
        *,
        owns_buffer: bool,
        axes_edges: EdgesLike = None,
        axes_nodes: EdgesLike = None,
    ):
        if self.closed:
            raise ClosedGraphError(
                "Unable to set input data.", node=self._node, input=self
            )
        if self.own_data is not None:
            raise AllocationError(
                "Input already has data.", node=self._node, input=self
            )

        self._own_data = data
        self._owns_buffer = owns_buffer
        self.own_dd.dtype = data.dtype
        self.own_dd.shape = data.shape
        self.own_dd.axes_edges = axes_edges
        self.own_dd.axes_nodes = axes_nodes

    @property
    def closed(self):
        return self._node.closed if self.node else False

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
        # NOTE: if the node is being evaluated, we must touch the node
        #       (trigger deep evaluation), else we get the data directly
        if self.node.being_evaluated:
            return self._parent_output.data
        return self._parent_output.get_data_unsafe()

    def get_data_unsafe(self):
        return self._parent_output.get_data_unsafe()

    @property
    def dd(self):
        return self._parent_output.dd

    @property
    def tainted(self) -> bool:
        return self._parent_output.tainted

    def touch(self):
        return self._parent_output.touch()

    def taint(self, **kwargs) -> None:
        self._node.taint(caller=self, **kwargs)

    def taint_type(self, *args, **kwargs) -> None:
        self._node.taint_type(*args, **kwargs)

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

        if self.own_dd.shape is None or self.own_dd.dtype is None:
            raise AllocationError(
                "No shape/type information provided for the Input",
                node=self._node,
                output=self,
            )
        try:
            self._own_data = zeros(self.own_dd.shape, self.own_dd.dtype, **kwargs)
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
