from __future__ import print_function

from .tools import StopNesting, undefinedname, undefinednode, undefinedoutput

from .edges import EdgeContainer
from .output import Output
from .shift import lshift, rshift
from .tools import IsIterable

# TODO: Why there are two outputs and how it works?


class Input:
    _name = undefinedname
    _node = undefinednode
    _output = undefinedoutput
    _corresponding_output = undefinedoutput

    def __init__(self, name, node, corresponding_output=undefinedoutput):
        self._name = name
        self._node = node
        self._corresponding_output = corresponding_output

    def __str__(self):
        return "->| {self._name}"

    def _set_output(self, output):
        if self._output:
            raise RuntimeError("Output is already connected to the input")

        self._output = output

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
    def output(self):
        return self._output

    @property
    def invalid(self):
        """Checks validity of the input data"""
        return self._output.invalid

    @invalid.setter
    def invalid(self, invalid):
        """Sets the validity of the current node"""
        self._node.invalid = invalid

    @property
    def corresponding_output(self):
        return self._corresponding_output

    @property
    def data(self):
        if not self._output:
            raise RuntimeError("May not read data from disconnected input")
        return self._output.data

    @property
    def datatype(self):
        return self._output.datatype

    @property
    def tainted(self):
        return self._output.tainted

    def touch(self):
        return self._output.touch()

    def taint(self, force=False):
        self._node.taint(force)

    def connected(self):
        return bool(self._output)

    def disconnected(self):
        return not bool(self._output)

    def _deep_iter_inputs(self, disconnected_only=False):
        if disconnected_only and self.connected():
            return iter(tuple())

        raise StopNesting(self)

    def _deep_iter_corresponding_outputs(self):
        if self._corresponding_output:
            raise StopNesting(self._corresponding_output)

        return iter(tuple())

    def __rshift__(self, other):
        """
        self >> other
        """
        if IsIterable(other):
            return rshift(self, other)
        self._set_output(
            other if isinstance(other, Output) else Output(other, self.node)
        )
        self.node.outputs += self._output
        return self._output

    def __lshift__(self, other):
        """
        self << other
        """
        if IsIterable(other):
            return lshift(self, other)

    def __rrshift__(self, other):
        """
        other >> self
        """
        if IsIterable(other):
            return lshift(self, other)


class Inputs(EdgeContainer):
    _datatype = Input

    def __init__(self, iterable=None):
        EdgeContainer.__init__(self, iterable)

    def __str__(self):
        return f"->[{len(self)}]|"

    def _deep_iter_inputs(self, disconnected_only=False):
        for input in self:
            if disconnected_only and input.connected():
                continue

            yield input

    def _touch(self):
        for input in self:
            input.touch()
