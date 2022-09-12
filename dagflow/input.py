from __future__ import annotations, print_function

from typing import TYPE_CHECKING, Iterator, Union

from .edges import EdgeContainer
from .output import Output
from .shift import lshift, rshift
from .tools import IsIterable, StopNesting, Undefined, undefined

if TYPE_CHECKING:
    from .node import Node


class Input:
    _closed: bool = False
    _allocated: bool = True
    _debug: bool = False

    def __init__(
        self,
        name: Union[str, Undefined] = undefined("name"),
        node: Union[Node, Undefined] = undefined("node"),
        iinput: Union[Input, Undefined] = undefined("iinput"),
        output: Union[Output, Undefined] = undefined("output"),
        **kwargs,
    ):
        self._name = name
        self._node = node
        self._iinput = iinput
        self._output = output
        self._closed = kwargs.pop("closed", False)
        self._debug = kwargs.pop("debug", node.debug if node else False)

    def __str__(self) -> str:
        return f"->| {self._name}"

    def __repr__(self) -> str:
        return self.__str__()

    def set_iinput(self, iinput: Input, force: bool = False) -> None:
        if self.closed:
            self.logger.warning(
                f"Input '{self.name}': "
                "A modification of the closed input is restricted!"
            )
        else:
            return self._set_iinput(iinput, force)

    def _set_iinput(self, iinput: Input, force: bool = False) -> None:
        self.logger.debug(
            f"Input '{self.name}': Adding iinput '{iinput.name}'..."
        )
        if self.iinput and not force:
            raise RuntimeError(
                f"The iinput is already setted to {self.iinput}!"
            )
        self._iinput = iinput

    def set_output(self, output: Output, force: bool = False) -> None:
        if self.closed:
            self.logger.warning(
                f"Input '{self.name}': "
                "A modification of the closed input is restricted!"
            )
        else:
            return self._set_output(output, force)

    def _set_output(self, output: Output, force: bool = False) -> None:
        self.logger.debug(
            f"Input '{self.name}': Adding output '{output.name}'..."
        )
        if self.connected() and not force:
            raise RuntimeError(
                f"The output is already setted to {self.output}!"
            )
        self._output = output

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name) -> None:
        self._name = name

    @property
    def node(self) -> Node:
        return self._node

    @property
    def logger(self):
        return self._node.logger

    @property
    def iinput(self) -> Input:
        return self._iinput

    @property
    def invalid(self) -> bool:
        """Checks validity of the output data"""
        return self._output.invalid

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def allocated(self) -> bool:
        return self._allocated

    @property
    def debug(self) -> bool:
        return self._debug

    @invalid.setter
    def invalid(self, invalid) -> None:
        """Sets the validity of the current node"""
        self._node.invalid = invalid

    @property
    def output(self) -> Output:
        return self._output

    def update_shape(self) -> None:
        if self._shape is undefined("shape"):
            self._shape = self.inputs[0].data.shape

    def update_dtype(self) -> None:
        if self._dtype is undefined("dtype"):
            self._dtype = self.inputs[0].data.dtype

    @property
    def data(self):
        if not self.connected():
            raise RuntimeError("May not read data from disconnected output!")
        return self._output.data

    @property
    def dtype(self):
        if not self.connected():
            raise RuntimeError(
                "May not read dtype from disconnected output!"
            )
        return self._output.dtype

    @property
    def shape(self):
        if not self.connected():
            raise RuntimeError(
                "May not read shape from disconnected output!"
            )
        return self._output.shape

    @property
    def tainted(self) -> bool:
        return self._output.tainted

    def touch(self):
        return self._output.touch()

    def taint(self, force: bool = False) -> None:
        self._node.taint(force)

    def connected(self) -> bool:
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

    def close(self, **kwargs) -> bool:
        self.logger.debug(f"Input '{self.name}': Closing input...")
        if self._closed:
            return True
        self._closed = True
        if self._iinput:
            self._closed = self._iinput.close(**kwargs)
            if not self._closed:
                self.logger.warning(
                    f"Input '{self.name}': The input is still open!"
                )
                return False
        # important
        if self.output:
            self._closed = self._output.close(**kwargs)
            if not self._closed:
                self.logger.warning(
                    f"Input '{self.name}': The output is still open!"
                )
                return False
            self.allocate(**kwargs)
        return self._closed

    def allocate(self, **kwargs) -> bool:
        self.logger.debug(f"Input '{self.name}': Allocating memory...")
        if self._allocated:
            self.logger.warning(f"Input '{self.name}': Memory is already allocated!")
            return True
        self._allocated = True
        if self._iinput:
            self._allocated = self._iinput.allocate(**kwargs)
            if not self._allocated:
                self.logger.warning(
                    f"Input '{self.name}': "
                    "The input memory is not allocated!"
                )
                return False
        # important
        if self.output:
            self._allocated = self._output.allocate(**kwargs)
            if not self._allocated:
                self.logger.warning(
                    f"Input '{self.name}': "
                    "The output memory is not allocated!"
                )
        return self._allocated

    def open(self) -> bool:
        self.logger.debug(f"Input '{self.name}': Opening the input...")
        if not self._closed:
            return True
        self._closed = False
        if self._iinput:
            self._closed = not self._iinput.open()
            if self._closed:
                self.logger.warning(
                    f"Input '{self.name}': The input is still closed!"
                )
        return not self._closed


class Inputs(EdgeContainer):
    _dtype = Input
    _closed: bool = False

    def __init__(self, iterable=None):
        super().__init__(iterable)

    def __str__(self):
        return f"->[{tuple(obj.name for obj in self)}]|"

    def _deep_iter_inputs(
        self, disconnected_only: bool = False
    ) -> Iterator[Input]:
        for input in self:
            if disconnected_only and input.connected():
                continue
            yield input

    def _deep_iter_iinputs(self) -> Iterator[Union[Input, Output]]:
        for iinput in self:
            yield iinput.iinput

    def _touch(self) -> None:
        for input in self:
            input.touch()

    def close(self) -> bool:
        if self._closed:
            return True
        self._closed = all(inp.close() for inp in self)
        return self._closed

    def open(self) -> bool:
        if not self._closed:
            return True
        self._closed = not all(inp.open() for inp in self)
        return not self._closed
