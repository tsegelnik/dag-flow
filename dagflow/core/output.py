from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from numpy import zeros

from ..core.labels import Labels, repr_pretty
from .data_descriptor import DataDescriptor
from .edges import EdgeContainer
from .exception import (
    AllocationError,
    CalculationError,
    ClosedGraphError,
    ConnectionError,
    DagflowError,
    InitializationError,
    UnclosedGraphError,
)
from .iter import StopNesting
from .shift import rshift

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from multikeydict.nestedmkdict import NestedMKDict

    from .input import Input
    from .node import Node
    from .types import EdgesLike, ShapeLike


class Output:
    __slots__ = (
        "_data",
        "_data_ro",
        "_dd",
        "_node",
        "_name",
        "_child_inputs",
        "_parent_input",
        "_allocating_input",
        "_allocatable",
        "_owns_buffer",
        "_forbid_reallocation",
        "_labels",
        "_debug",
    )
    _data: NDArray | None
    _data_ro: NDArray | None
    _dd: DataDescriptor
    _labels: Labels | None
    _node: Node | None
    _name: str | None

    _child_inputs: list[Input]
    _parent_input: Input | None
    _allocating_input: Input | None

    _allocatable: bool
    _owns_buffer: bool
    _forbid_reallocation: bool
    _debug: bool

    def __init__(
        self,
        name: str | None,
        node: Node | None,
        *,
        debug: bool | None = None,
        allocatable: bool | None = None,
        data: NDArray | None = None,
        owns_buffer: bool | None = None,
        dtype: DTypeLike = None,
        shape: ShapeLike | None = None,
        axes_edges: EdgesLike | None = None,
        axes_meshes: EdgesLike | None = None,
        forbid_reallocation: bool = False,
    ):
        self._labels = None
        self._data = None
        self._data_ro = None
        self._allocating_input = None

        self._name = name
        self._node = node
        self._child_inputs = []
        self._debug = debug if debug is not None else node.debug if node else False
        self._forbid_reallocation = forbid_reallocation

        self._dd = DataDescriptor(dtype, shape, axes_edges, axes_meshes)

        if data is None:
            self._allocatable = True if allocatable is None else allocatable
            self._parent_input = None
            self._owns_buffer = False
        else:
            if owns_buffer is None:
                owns_buffer = True
            self._allocatable = not owns_buffer
            self._set_data(data, owns_buffer=owns_buffer)

            if allocatable or dtype is not None or shape is not None:
                raise InitializationError(output=self, node=node)

    def __str__(self):
        return self.connected() and f"●→ {self._name}" or f"○→ {self._name}"

    _repr_pretty_ = repr_pretty

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
    def labels(self) -> Labels | None:
        return self._labels if self._labels is not None else self._node and self._node.labels

    @labels.setter
    def labels(self, labels: Labels):
        self._labels = labels

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
    def data(self) -> NDArray:
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
            return self._data_ro
        except CalculationError as exc:
            raise CalculationError(
                "An exception occured while the node was touched!",
                node=self._node,
                output=self,
                args=exc.args,
            ) from exc

    def _set_data(
        self,
        data,
        *,
        owns_buffer: bool,
        override: bool = False,
        forbid_reallocation: bool | None = None,
    ):
        if self.closed:
            raise ClosedGraphError("Unable to set output data.", node=self._node, output=self)
        # if self._data is not None and not override:
        #     # NOTE: this will fail during reallocation
        #     raise AllocationError("Output already has data.", node=self._node, output=self)
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

        if not self.dd.consistent_with(data):
            self.node.taint_type()

        self._data = data
        self._data_ro = data.view()
        self._data_ro.flags.writeable = False

        self.dd.dtype = data.dtype
        self.dd.shape = data.shape
        self._owns_buffer = owns_buffer
        self._forbid_reallocation = forbid_reallocation

    @property
    def dd(self) -> DataDescriptor:
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

    @property
    def data_unsafe(self):
        return self._data

    def connect_to(self, input) -> Input:
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

    def _connect_to(self, input) -> Input:
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
        input.set_parent_output(self)
        return input

    def deep_iter_outputs(self, disconnected_only=False):
        if disconnected_only and self.connected():
            return iter(tuple())
        raise StopNesting(self)

    def deep_iter_child_outputs(self):
        raise StopNesting(self)

    def __rshift__(
        self,
        other: Input | Node | Sequence[Input] | Sequence[Node] | Mapping[str, Node] | NestedMKDict,
    ):
        """
        self >> other
        """
        from multikeydict.nestedmkdict import NestedMKDict

        from .input import Input
        from .node import Node

        if isinstance(other, Input):
            self.connect_to(other)
        elif isinstance(other, Node):
            rshift(self, other)
        elif isinstance(other, Sequence):
            for subother in other:
                self >> subother
        elif isinstance(other, Mapping):
            for subother in other.values():
                self >> subother
        elif isinstance(other, NestedMKDict):
            for subother in other.walkvalues():
                self >> subother
        else:
            rshift(self, other)

    def taint_children(self, *, force: bool = False, force_computation: bool = False, caller: Input| None=None) -> None:
        for input in self._child_inputs:
            input.taint(force=force, force_computation=force_computation, caller=caller)

    def taint_children_type(self, **kwargs) -> None:
        for input in self._child_inputs:
            input.taint_type(**kwargs)

    def touch(self, force_computation=False):
        return self._node.touch(force_computation=force_computation)

    def connected(self):
        return bool(self._child_inputs)

    def allocate(self, **kwargs) -> bool:
        """returns True if data was reassigned"""
        if not self._allocatable:
            return False

        if self._allocating_input:
            _input = self._allocating_input
            _input.allocate(recursive=False)
            if _input.has_data:
                idata = _input._own_data
                if not self.dd.consistent_with(idata):
                    raise AllocationError(
                        "Input's data shape/type is inconsistent",
                        node=self._node,
                        output=self,
                        input=_input,
                    )

                if self._data is not idata:
                    if self._data is not None:
                        idata[:] = self._data
                    self._set_data(idata, owns_buffer=False, override=True)
                return True

        if (data := self._data) is not None and self.dd.consistent_with(data):
            return False

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
            raise AllocationError(f"Output: {exc.args[0]}", node=self._node, output=self) from exc

        return True

    def seti(self, idx: int, value: float, check_taint: bool = False, force: bool = False) -> bool:
        if self.node.frozen and not force:
            return False

        tainted = self._data[idx] != value if check_taint else True
        if tainted:
            self._data[idx] = value
            self.__taint_children()
        return tainted

    def set(self, data: ArrayLike, check_taint: bool = False, force: bool = False) -> bool:
        if self.node.frozen and not force:
            return False

        tainted = (self._data != data).any() if check_taint else True
        if tainted:
            self._data[:] = data
            self.__taint_children()
        return tainted

    # TODO: maybe move it into `self.taint_children()`?
    def __taint_children(self):
        self.taint_children()
        self.node.invalidate_parents()
        self.node.fd.tainted = False

    def to_dict(self, *, label_from: str = "text") -> dict:
        shape = self.dd.shape
        size = self.dd.size
        ret = {
            "label": self.labels[label_from],
            "shape": shape[0] if shape and len(shape) == 1 else shape,
        }

        if size is not None:
            if size > 1:
                ret["value"] = "…"
            elif size == 1:
                try:
                    data = self.data
                except DagflowError:
                    ret["value"] = "???"
                else:
                    ret["value"] = float(data.ravel()[0])

        return ret


class Outputs(EdgeContainer):
    __slots__ = ()

    def __init__(self, iterable=None) -> None:
        super().__init__(iterable)
        self._dtype = Output

    def __str__(self) -> str:
        return f"○[{tuple(obj.name for obj in self)}]→"

    _repr_pretty_ = repr_pretty
