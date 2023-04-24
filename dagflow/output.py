from itertools import cycle
from typing import List, Optional, Tuple

from numpy import zeros
from numpy.typing import ArrayLike, DTypeLike, NDArray

from .edges import EdgeContainer
from .exception import (
    ClosedGraphError,
    CriticalError,
    InitializationError,
    AllocationError,
    ConnectionError,
    UnclosedGraphError,
)
from .shift import lshift, rshift
from .iter import StopNesting
from .types import EdgesLike, InputT, NodeT, ShapeLike
from .datadescriptor import DataDescriptor


class Output:
    _data: Optional[NDArray] = None
    _dd: DataDescriptor

    _node: Optional[NodeT]
    _name: Optional[str]

    _child_inputs: List[InputT]
    _parent_input: Optional[InputT] = None
    _allocating_input: Optional[InputT] = None

    _allocatable: bool = True
    _owns_buffer: bool = False
    _forbid_reallocation: bool = False

    _debug: bool = False

    def __init__(
        self,
        name: Optional[str],
        node: Optional[NodeT],
        *,
        debug: Optional[bool] = None,
        allocatable: Optional[bool] = None,
        data: Optional[NDArray] = None,
        owns_buffer: Optional[bool] = None,
        dtype: DTypeLike = None,
        shape: Optional[ShapeLike] = None,
        axes_edges: Optional[Tuple[EdgesLike]] = None,
        axes_nodes: Optional[Tuple[EdgesLike]] = None,
        forbid_reallocation: bool = False,
    ):
        self._name = name
        self._node = node
        self._child_inputs = []
        self._debug = (
            debug if debug is not None else node.debug if node else False
        )
        self._forbid_reallocation = forbid_reallocation

        self._dd = DataDescriptor(dtype, shape, axes_edges, axes_nodes)

        if data is None:
            self._allocatable = True if allocatable is None else allocatable
        else:
            if owns_buffer is None:
                owns_buffer = True
            self._allocatable = not owns_buffer
            self._set_data(data, owns_buffer=owns_buffer)

            if allocatable or dtype is not None or shape is not None:
                raise InitializationError(output=self, node=node)

    def __str__(self):
        return f"●→ {self._name}" if self.owns_buffer else f"○→ {self._name}"

    def __repr__(self):
        return self.__str__()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def allocatable(self):
        return self._allocatable

    @property
    def has_data(self) -> bool:
        return self._data is not None

    @property
    def node(self):
        return self._node

    @property
    def child_inputs(self):
        return self._child_inputs

    @property
    def parent_input(self):
        return self._parent_input

    @parent_input.setter
    def parent_input(self, input):
        self._parent_input = input

    @property
    def logger(self):
        return self._node.logger

    @property
    def invalid(self):
        """Checks the validity of the current node"""
        return self._node.invalid

    @invalid.setter
    def invalid(self, invalid):
        """Sets the validity of the following nodes"""
        for input in self.child_inputs:
            input.invalid = invalid

    @property
    def data(self):
        if self.node.being_evaluated:
            return self._data
        if not self.closed:
            raise UnclosedGraphError(
                "Unable to get the output data from unclosed graph!",
                node=self._node,
                output=self,
            )
        try:
            self.touch()
            return self.get_data_unsafe()
        except Exception as exc:
            raise CriticalError(
                "An exception occured during touching of the parent node!",
                node=self._node,
                output=self,
            ) from exc

    def _set_data(
        self,
        data,
        *,
        owns_buffer: bool,
        override: bool = False,
        forbid_reallocation: Optional[bool] = None,
    ):
        if self.closed:
            raise ClosedGraphError(
                "Unable to set output data.", node=self._node, output=self
            )
        if self._data is not None and not override:
            # TODO: this will fail during reallocation
            raise AllocationError(
                "Output already has data.", node=self._node, output=self
            )
        if owns_buffer:
            forbid_reallocation = True
        elif forbid_reallocation is None:
            forbid_reallocation = owns_buffer

        forbid_reallocation |= self._forbid_reallocation
        if forbid_reallocation and self._allocating_input:
            raise AllocationError(
                "Output is connected to allocating input, but reallocation is forbidden",
                node=self._node,
                output=self,
            )

        self._data = data
        self.dd.dtype = data.dtype
        self.dd.shape = data.shape
        self._owns_buffer = owns_buffer
        self._forbid_reallocation = forbid_reallocation

    @property
    def dd(self) -> Optional[DataDescriptor]:
        return self._dd

    @property
    def owns_buffer(self):
        return self._owns_buffer

    @property
    def forbid_reallocation(self):
        return self._forbid_reallocation

    @property
    def closed(self):
        return self.node.closed if self.node else False

    @property
    def tainted(self) -> bool:
        return self._node.tainted

    @property
    def debug(self) -> bool:
        return self._debug

    def get_data_unsafe(self):
        return self._data

    def connect_to(self, input) -> InputT:
        if not self.closed and input.closed:
            raise ConnectionError(
                "Cannot connect an output to a closed input!",
                node=self.node,
                output=self,
                input=input,
            )
        if self.closed and input.allocatable:
            raise ConnectionError(
                "Cannot connect a closed output to an allocatable input!",
                node=self.node,
                output=self,
                input=input,
            )
        return self._connect_to(input)

    def _connect_to(self, input) -> InputT:
        if input.allocatable:
            if self._allocating_input:
                raise ConnectionError(
                    "Output has multiple allocatable/allocated child inputs",
                    node=self._node,
                    output=self,
                )
            if self._forbid_reallocation:
                raise ConnectionError(
                    "Output forbids reallocation and may not connect to allocating inputs",
                    node=self._node,
                    output=self,
                )
            self._allocating_input = input
        self._child_inputs.append(input)
        input._set_parent_output(self)
        return input

    def __rshift__(self, other):
        return rshift(self, other)

    def __rlshift__(self, other):
        return lshift(self, other)

    def taint_children(self, **kwargs) -> None:
        for input in self._child_inputs:
            input.taint(**kwargs)

    def taint_children_type(self, **kwargs) -> None:
        for input in self._child_inputs:
            input.taint_type(**kwargs)

    def touch(self):
        return self._node.touch()

    def connected(self):
        return bool(self._child_inputs)

    def deep_iter_outputs(self, disconnected_only=False):
        if disconnected_only and self.connected():
            return iter(tuple())
        raise StopNesting(self)

    def deep_iter_child_outputs(self):
        raise StopNesting(self)

    def repeat(self):
        return RepeatedOutput(self)

    def allocate(self, **kwargs):
        if not self._allocatable:
            return True

        if self._allocating_input:
            input = self._allocating_input
            input.allocate(recursive=False)
            if input.has_data:
                idata = input._own_data
                if idata.shape != self.dd.shape or idata.dtype != self.dd.dtype:
                    raise AllocationError(
                        "Input's data shape/type is inconsistent",
                        node=self._node,
                        output=self,
                        input=input,
                    )

                if self._data is not idata:
                    if self._data is not None:
                        idata[:] = self._data
                    self._set_data(idata, owns_buffer=False, override=True)
                return True

        if self.has_data:
            return True

        if self.dd.shape is None or self.dd.dtype is None:
            raise AllocationError(
                "No shape/type information provided for the Output",
                node=self._node,
                output=self,
            )
        try:
            data = zeros(self.dd.shape, self.dd.dtype, **kwargs)
            self._set_data(data, owns_buffer=True)
        except Exception as exc:
            raise AllocationError(
                f"Output: {exc.args[0]}", node=self._node, output=self
            ) from exc

        return True

    def seti(self, idx: int, value: float, check_taint: bool = False, force: bool = False) -> bool:
        if self.node._frozen and not force:
            return False

        tainted = True
        if check_taint:
            tainted = self._data[udx] != value

        if tainted:
            self._data[idx] = value
            self.taint_children()
            self.node.invalidate_parents()
            self.node._tainted = False

        return tainted

    def set(
        self, data: ArrayLike, check_taint: bool = False, force: bool = False
    ) -> bool:
        if self.node._frozen and not force:
            return False

        tainted = True
        if check_taint:
            tainted = (self._data != data).any()

        if tainted:
            self._data[:] = data
            self.taint_children()
            self.node.invalidate_parents()
            self.node._tainted = False

        return tainted


class RepeatedOutput:
    def __init__(self, output):
        self._output = output

    def __iter__(self):
        return cycle((self._output,))

    def __rshift__(self, other):
        return rshift(self, other)

    def __rlshift__(self, other):
        return lshift(self, other)


class Outputs(EdgeContainer):
    _dtype = Output

    def __init__(self, iterable=None) -> None:
        super().__init__(iterable)

    def __str__(self) -> str:
        return f"○[{tuple(obj.name for obj in self)}]→"

    def __repr__(self) -> str:
        return self.__str__()
