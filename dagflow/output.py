from itertools import cycle
from typing import List, Optional, Tuple

from numpy import zeros
from numpy.typing import ArrayLike, DTypeLike, NDArray

from .edges import EdgeContainer
from .exception import (
    ClosedGraphError,
    InitializationError,
    AllocationError,
    CriticalError,
    ConnectionError
)
from .shift import lshift, rshift
from .tools import StopNesting
from .types import InputT, NodeT


class Output:
    _data: Optional[NDArray] = None
    _dtype: Optional[DTypeLike] = None
    _shape: Optional[Tuple[int, ...]] = None
    _owns_data: bool = False

    _node: Optional[NodeT]
    _name: Optional[str]

    _child_inputs: List[InputT]
    _parent_input: Optional[InputT] = None

    _allocatable: bool = True
    _allocated: bool = False
    _debug: bool = False

    def __init__(
        self,
        name: Optional[str],
        node: Optional[NodeT],
        *,
        debug: Optional[bool] = None,
        allocatable: bool = True,
        owns_data: bool = True,
        data: Optional[NDArray] = None,
        dtype: Optional[DTypeLike] = None,
        shape: Optional[Tuple[int, ...]] = None,
    ):
        self._name = name
        self._node = node
        self._child_inputs = []
        self._debug = (
            debug if debug is not None else node.debug if node else False
        )
        self._allocatable = allocatable
        self._dtype = dtype
        self._shape = shape
        if data is not None and (
            allocatable or dtype is not None or shape is not None
        ):
            raise InitializationError(output=self, node=node)

        if data is not None:
            self.data = data
            self._owns_data = owns_data

    def __str__(self):
        return f"●→ {self._name}" if self.owns_data else f"○→ {self._name}"

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
    def allocated(self):
        return self._allocated

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
        if not self.evaluated:
            self.touch()
        return self._data

    @data.setter
    def data(self, data):
        if self.closed:
            raise ClosedGraphError(
                "Unable to set output data.", node=self._node, output=self
            )
        if self._data is not None:
            raise AllocationError(
                "Output already has data.", node=self._node, output=self
            )

        self._data = data
        self._dtype = data.dtype
        self._shape = data.shape
        self._allocated = True

    @property
    def owns_data(self):
        return self._owns_data

    @property
    def closed(self):
        return self.node.closed if self.node else False

    @property
    def evaluated(self):
        return self.node.evaluated if self.node else False

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape

    @property
    def tainted(self) -> bool:
        return self._node.tainted

    @property
    def debug(self) -> bool:
        return self._debug

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
        self._child_inputs.append(input)
        input._set_parent_output(self)
        return input

    def __rshift__(self, other):
        return rshift(self, other)

    def __rlshift__(self, other):
        return lshift(self, other)

    def taint_children(self, force=False):
        for input in self._child_inputs:
            input.taint(force)

    def taint(self, force=False):
        for input in self._child_inputs:
            input.taint(force)

    def taint_type(self, force=False):
        for input in self._child_inputs:
            input.taint_type(force)

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
        if not self._allocatable or self._allocated:
            return True

        if len(self._child_inputs) == 1:
            input = self._child_inputs[0]
            input.allocate(recursive=False)
            if input.allocated:
                idata = input._own_data
                if idata.shape != self.shape or idata.dtype != self.dtype:
                    raise AllocationError(
                        "Input's data shape/type is inconsistent",
                        node=self._node,
                        output=self,
                        input=input,
                    )

                self.data = idata
                return True
        elif any(
            input.allocated or input.allocatable
            for input in self._child_inputs
        ):
            raise AllocationError(
                "Output has multiple allocatable/allocated child inputs",
                node=self._node,
                output=self,
            )

        if self.shape is None or self.dtype is None:
            raise AllocationError(
                "No shape/type information provided for the Output",
                node=self._node,
                output=self,
            )
        try:
            self._data = zeros(self.shape, self.dtype, **kwargs)
        except Exception as exc:
            raise AllocationError(
                f"Output: {exc.args[0]}", node=self._node, output=self
            ) from exc

        self._owns_data = True
        self._allocated = True
        return True

class SettableOutput(Output):
    def set(self, data: ArrayLike, check_taint: bool=False, force: bool=False) -> bool:
        if self.node._frozen and not force:
            return False

        tainted = True
        if check_taint:
            tainted = (self._data!=data).any()

        if tainted:
            self._data[:]=data
            self.taint()
            self.node.invalidate_parents()
            self.node._tainted=False

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
