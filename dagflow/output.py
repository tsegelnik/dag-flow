
from itertools import cycle
from typing import Optional, List

from numpy import zeros, ndarray
from numpy.typing import ArrayLike

from .edges import EdgeContainer
from .shift import lshift, rshift
from .tools import StopNesting, undefined
from .types import InputT

class Output:
    _name = undefined("name")
    _node = undefined("node")
    _child_inputs: List[InputT]
    _parent_input: Optional[InputT] = None
    _data: ndarray = undefined("data")
    _dtype = undefined("dtype")
    _shape = undefined("shape")
    _allocatable: bool = True
    _allocated: bool = False
    _closed: bool = False
    _debug: bool = False

    def __init__(self, name, node, *, allocatable: bool=True, data: ArrayLike=None, debug: Optional[bool]=None):
        self._name = name
        self._node = node
        self._child_inputs = []
        self._debug = debug if debug is not None else node.debug if node else False
        self._allocatable = allocatable
        if not self._allocatable:
            if data is not None:
                self.data = data
            self._allocated = True

    def __str__(self):
        return f"|-> {self._name}"

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
    def data(self, val):
        if self._data is undefined("data"):
            self._data = val
            self._dtype = val.dtype
            self._shape = val.shape
            self._allocated = True
        else:
            self.logger.warning(
                f"Output '{self.name}': The data is already set!"
            )

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
    def closed(self) -> bool:
        return self._closed

    @property
    def debug(self) -> bool:
        return self._debug

    def view(self, **kwargs):
        if self._allocated:
            return self.data.view(**kwargs)
        self.logger.warning(
            f"Output '{self.name}': The output memory is not allocated."
        )

    def connect_to(self, input):
        if not self.closed:
            return self._connect_to(input)
        self.logger.warning(
            f"Output '{self.name}': "
            "A modification of the closed output is restricted!"
        )

    def _connect_to(self, input):
        if input in self._child_inputs:
            raise RuntimeError(
                f"Output '{self.name}' is already connected "
                f"to the input '{input.name}'!"
            )
        self._child_inputs.append(input)
        input._set_parent_output(self)
        return input

    def __rshift__(self, other):
        return rshift(self, other)

    def __rlshift__(self, other):
        return lshift(self, other)

    def taint(self, force=False):
        for input in self._child_inputs:
            input.taint(force)

    def touch(self):
        return self._node.touch()

    def connected(self):
        return bool(self._child_inputs)

    def _deep_iter_outputs(self, disconnected_only=False):
        if disconnected_only and self.connected():
            return iter(tuple())
        raise StopNesting(self)

    def _deep_iter_child_outputs(self):
        raise StopNesting(self)

    def repeat(self):
        return RepeatedOutput(self)

    def allocate(self, **kwargs):
        if not self._allocatable:
            self.logger.debug(
                f"Output '{self.name}': "
                f"The output is not allocatable: data={self._data}!"
            )
            return self._allocated
        if self._allocated:
            self.logger.debug(
                f"Output '{self.name}': "
                f"The output memory is already allocated: data={self._data}!"
            )
            return self._allocated
        self.logger.info(f"Output '{self.name}': Allocate the memory...")
        try:
            # NOTE: may be troubles with multidimensional arrays!
            self._data = zeros(self.shape, self.dtype, **kwargs)
            self.logger.info(
                f"Output '{self.name}': The memory is successfully allocated!"
            )
            self._allocated = True
        except Exception as exc:
            self.logger.error(
                f"Output '{self.name}': The output memory is not allocated "
                f"due to the exception: {exc}!"
            )
            self._allocated = False
        return self._allocated

    def _close(self, **kwargs) -> bool:
        self.logger.debug(f"Output '{self.name}': Closing output...")
        if self._closed:
            return True
        self._closed = self._allocated
        if not self._closed:
            self.logger.warning(
                f"Output '{self.name}': Some child inputs are still open: "
                f"'{tuple(inp.name for inp in self._child_inputs if inp.closed)}'!"
            )
        else:
            self.logger.debug(
                f"Output '{self.name}': The closure completed successfully!"
            )
        return self.closed

    def close(self, **kwargs) -> bool:
        self.logger.debug(f"Output '{self.name}': Closing output...")
        if self._closed:
            return True
        self._closed = all(inp.close(**kwargs) for inp in self._child_inputs)
        if not self._closed:
            self.logger.warning(
                "Output '{self.name}': Some child inputs are still open: "
                f"'{tuple(inp.name for inp in self._child_inputs if inp.closed)}'!"
            )
            return False
        self._closed = self.node.close(**kwargs)
        if not self._closed:
            self.logger.warning(
                f"Output '{self.name}': "
                f"The node '{self.node}' is still open!"
            )
            return False
        if self.allocatable:
            self.allocate(**kwargs)
        return self.closed

    def open(self) -> bool:
        self.logger.debug(f"Output '{self.name}': Opening output...")
        if not self._closed:
            return True
        self._closed = not all(inp.open() for inp in self._child_inputs)
        if self._closed:
            self.logger.warning(
                f"Output '{self.name}': Some child inputs are still closed: "
                f"'{tuple(inp.name for inp in self._child_inputs if inp.closed)}'!"
            )
        return not self._closed


class RepeatedOutput:
    _closed: bool = False

    def __init__(self, output):
        self._output = output

    @property
    def closed(self) -> bool:
        return self._closed

    def __iter__(self):
        return cycle((self._output,))

    def __rshift__(self, other):
        return rshift(self, other)

    def __rlshift__(self, other):
        return lshift(self, other)

    def close(self) -> bool:
        if self._closed:
            return True
        self._closed = self._output.close()
        if not self._closed:
            self.logger.warning(
                f"Output '{self.name}': The output is still open!"
            )
        return self._closed

    def open(self) -> bool:
        if not self._closed:
            return True
        self._closed = not self._output.open()
        if self._closed:
            self.logger.warning(
                f"Output '{self.name}': The output is still closed!"
            )
        return not self._closed


class Outputs(EdgeContainer):
    _dtype = Output

    def __init__(self, iterable=None) -> None:
        super().__init__(iterable)

    def __str__(self) -> str:
        return f"|[{tuple(obj.name for obj in self)}]->"

    def __repr__(self) -> str:
        return self.__str__()

    def close(self) -> bool:
        if self._closed:
            return True
        self._closed = all(out.close() for out in self)
        return self._closed

    def open(self) -> bool:
        if not self._closed:
            return True
        self._closed = not all(out.open() for out in self)
        return not self._closed
