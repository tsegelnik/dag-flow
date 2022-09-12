from __future__ import print_function

from itertools import cycle

from numpy import zeros, result_type

from .edges import EdgeContainer
from .shift import lshift, rshift
from .tools import StopNesting, undefined


class Output:
    _name = undefined("name")
    _node = undefined("node")
    _inputs = None
    _data = undefined("data")
    _dtype = undefined("dtype")
    _shape = undefined("shape")
    _allocated: bool = False
    _closed: bool = False
    _debug: bool = False

    def __init__(self, name, node, **kwargs):
        self._name = name
        self._node = node
        self._inputs = []
        self._debug = kwargs.pop("debug", node.debug if node else False)

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
    def allocated(self):
        return self._allocated

    @property
    def node(self):
        return self._node

    @property
    def inputs(self):
        return self._inputs

    @property
    def invalid(self):
        """Checks the validity of the current node"""
        return self._node.invalid

    @invalid.setter
    def invalid(self, invalid):
        """Sets the validity of the following nodes"""
        for input in self.inputs:
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
            print(f"WARNING: Output '{self.name}': The data is already set!")

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

    def update_shape(self) -> None:
        # TODO: Custom method
        # raise Exception()
        if self._shape is undefined("shape"):
            if self.node and len(self.node.inputs) != 0:
                self._shape = self.node.inputs[0].shape
            elif len(self.inputs) != 0:
                self._shape = self.inputs[0].shape

    def update_dtype(self) -> None:
        # TODO: Custom method
        # raise Exception()
        if self._dtype is undefined("dtype"):
            if self.node and len(self.node.inputs) != 0:
                self._dtype = result_type(*self.node.inputs)
            elif len(self.inputs) != 0:
                self._dtype = result_type(*self.inputs)

    def view(self, dtype=None, type=None):
        if self._allocated:
            return self.data.view(dtype=dtype, type=type)
        print(
            f"WARNING: Output '{self.name}': "
            "The output memory is not allocated!"
        )

    def allocate(self, **kwargs):
        if self._allocated:
            print(
                f"WARNING: Output '{self.name}': "
                f"The output memory is already allocated: {self.data}!"
            )
            return self._allocated
        if self.debug:
            print(f"DEBUG: Output '{self.name}': Allocate the memory...")
        try:
            self.update_dtype()
            self.update_shape()
            self.data = zeros(self.shape, self.dtype, **kwargs)
            self._allocated = True
        except Exception as exc:
            print(
                f"WARNING: Output '{self.name}': "
                f"The output memory is not allocated due to exception '{exc}'!"
            )
            self._allocated = False

    def connect_to(self, input):
        if self.closed:
            print(
                f"WARNING: Output '{self.name}': "
                "A modification of the closed output is restricted!"
            )
        else:
            return self._connect_to(input)

    def _connect_to(self, input):
        if input in self._inputs:
            raise RuntimeError(
                f"Output '{self.name}' is already connected to the input '{input.name}'!"
            )
        self._inputs.append(input)
        input._set_output(self)
        return input

    def __rshift__(self, other):
        return rshift(self, other)

    def __rlshift__(self, other):
        return lshift(self, other)

    def taint(self, force=False):
        for input in self._inputs:
            input.taint(force)

    def touch(self):
        return self._node.touch()

    def connected(self):
        return bool(self._inputs)

    def _deep_iter_outputs(self, disconnected_only=False):
        if disconnected_only and self.connected():
            return iter(tuple())
        raise StopNesting(self)

    def _deep_iter_iinputs(self):
        raise StopNesting(self)

    def repeat(self):
        return RepeatedOutput(self)

    def close(self) -> bool:
        if self.debug:
            print(f"DEBUG: Output '{self.name}': Closing output...")
        if self._closed:
            return True
        self._closed = all(inp.close() for inp in self._inputs)
        if not self._closed:
            print(
                "WARNING: Output '{self.name}': Some inputs are still open: "
                f"'{tuple(inp.name for inp in self._inputs if inp.closed)}'!"
            )
            return False
        self._closed = self.node.close()
        if not self._closed:
            print(
                f"WARNING: Output '{self.name}': "
                f"The node '{self.node}' is still open!"
            )
        self.allocate()
        return self.closed

    def open(self) -> bool:
        if self.debug:
            print(f"DEBUG: Output '{self.name}': Opening output...")
        if not self._closed:
            return True
        self._closed = not all(inp.open() for inp in self._inputs)
        if self._closed:
            print(
                "WARNING: Output '{self.name}': Some inputs are still closed: "
                f"'{tuple(inp.name for inp in self._inputs if inp.closed)}'!"
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
            print(f"WARNING: Output '{self.name}': The output is still open!")
        return self._closed

    def open(self) -> bool:
        if not self._closed:
            return True
        self._closed = not self._output.open()
        if self._closed:
            print(
                f"WARNING: Output '{self.name}': The output is still closed!"
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
