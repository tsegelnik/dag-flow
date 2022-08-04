from __future__ import print_function

from itertools import cycle

from .edges import EdgeContainer
from .shift import lshift, rshift
from .tools import StopNesting, undefined


class Output:
    _name = undefined("name")
    _node = undefined("node")
    _inputs = None
    _data = undefined("data")
    _datatype = undefined("datatype")
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
        self._node.touch()
        return self._data

    @data.setter
    def data(self, data):
        if self._datatype is undefined("datatype"):
            self._datatype = type(data)
        elif self._datatype != type(data):
            raise TypeError("Unable to change existing data type")
        self._data = data
        return data

    @property
    def datatype(self):
        if self._datatype is undefined("datatype"):
            self._datatype = type(self.data)
        return self._datatype

    @property
    def tainted(self) -> bool:
        return self._node.tainted

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def debug(self) -> bool:
        return self._debug

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
    _datatype = Output

    def __init__(self, iterable=None) -> None:
        super().__init__(iterable)

    def __str__(self) -> str:
        # return f"|[{len(self)}]->"
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
