from __future__ import print_function

from itertools import cycle

from .edges import EdgeContainer
from .shift import lshift, rshift
from .tools import (
    StopNesting,
    undefineddata,
    undefineddatatype,
    undefinedname,
    undefinednode,
)


class Output:
    _name = undefinedname
    _node = undefinednode
    _inputs = None

    _data = undefineddata
    _datatype = undefineddatatype

    def __init__(self, name, node):
        self._name = name
        self._node = node
        self._inputs = []

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
        if self._datatype is undefineddatatype:
            self._datatype = type(data)
        elif self._datatype != type(data):
            raise TypeError("Unable to change existing data type")
        self._data = data
        return data

    @property
    def datatype(self):
        return self._datatype

    @property
    def tainted(self):
        return self._node.tainted

    def _connect_to(self, input):
        if input in self._inputs:
            raise RuntimeError("Output is already connected to the input")

        self._inputs.append(input)
        input._set_output(self)
        return input

    __rshift__ = rshift
    __rlshift__ = lshift

    #def __rshift__(self, other):
    #    from .input import Input
    #    from .node import Node
    #    if isinstance(other, Input):
    #        return self._connect_to(other)
    #    elif isinstance(other, Node):
    #        return other._add_input(self)
    #    return rshift(self, other)

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


class RepeatedOutput:
    def __init__(self, output):
        self._output = output

    def __iter__(self):
        return cycle((self._output,))

    __rshift__ = rshift
    __rlshift__ = lshift


class Outputs(EdgeContainer):
    _datatype = Output

    def __init__(self, iterable=None) -> None:
        super().__init__(iterable)

    def __str__(self) -> str:
        return f"|[{len(self)}]->"

    def __repr__(self) -> str:
        return self.__str__()
