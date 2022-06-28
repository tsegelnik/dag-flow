from __future__ import annotations, print_function

from typing import TYPE_CHECKING, Union

from .edges import EdgeContainer
from .output import Output
from .shift import lshift, rshift
from .tools import IsIterable, StopNesting, Undefined

if TYPE_CHECKING:
    from .node import Node


class Input:
    def __init__(
        self,
        name: Union[str, Undefined] = Undefined("name"),
        node: Union[Node, Undefined] = Undefined("Node"),
        iinput: Union[Input, Undefined] = Undefined("iinput"),
        output: Union[Output, Undefined] = Undefined("output"),
    ):
        self._name = name
        self._node = node
        self._iinput = iinput
        self._output = output

    def __str__(self):
        return f"->| {self._name}"

    def __repr__(self):
        return self.__str__()

    def _set_iinput(self, iinput: Input, force: bool = False):
        if self.iinput and not force:
            raise RuntimeError(
                f"The iinput is already setted to {self.iinput}!"
            )
        self._iinput = iinput

    def _set_output(self, output: Output, force: bool = False):
        if self.connected() and not force:
            raise RuntimeError(
                f"The output is already setted to {self.output}!"
            )
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
    def iinput(self):
        return self._iinput

    @property
    def invalid(self):
        """Checks validity of the output data"""
        return self._output.invalid

    @invalid.setter
    def invalid(self, invalid):
        """Sets the validity of the current node"""
        self._node.invalid = invalid

    @property
    def output(self):
        return self._output

    @property
    def data(self):
        if not self.connected():
            raise RuntimeError("May not read data from disconnected output!")
        return self._output.data

    @property
    def datatype(self):
        if not self.connected():
            raise RuntimeError(
                "May not read datatype from disconnected output!"
            )
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

    def _deep_iter_inputs(self, disconnected_only=False):
        if disconnected_only and self.connected():
            return iter(tuple())
        raise StopNesting(self)

    def _deep_iter_iinputs(self):
        if self._iinput:
            raise StopNesting(self._iinput)
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


class Inputs(EdgeContainer):
    _datatype = Input

    def __init__(self, iterable=None):
        super().__init__(iterable)

    def __str__(self):
        return f"->[{len(self)}]|"

    def _deep_iter_inputs(self, disconnected_only=False):
        for input in self:
            if disconnected_only and input.connected():
                continue
            yield input

    def _deep_iter_iinputs(self):
        for iinput in self:
            yield iinput.iinput

    def _touch(self):
        for input in self:
            input.touch()
