from typing import TYPE_CHECKING, Optional

from ..nodes import FunctionNode
from ..typefunctions import copy_from_input_to_output

if TYPE_CHECKING:
    from ..input import Input
    from ..output import Output


class View(FunctionNode):
    """Creates a node with a single data output which is a view on the input"""

    __slots__ = (
        "_input",
        "_start",
        "_length",
    )
    _input: "Input"
    _start: int | None
    _length: int | None

    def __init__(
        self,
        name,
        output: Optional["Output"] = None,
        *,
        outname="view",
        start: int | None = None,
        length: int | None = None,
        **kwargs
    ) -> None:
        super().__init__(name, **kwargs)
        child_output = self._add_output(outname, allocatable=False, forbid_reallocation=True)
        self._input = self._add_input("input", child_output=child_output)
        self._start = start
        self._length = length

        if output is not None:
            output >> self._input
            if output.closed:
                self.close()

    def _fcn(self):
        self._input.data

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        copy_from_input_to_output(self, 0, 0)

    def _post_allocate(self) -> None:
        _input = self.inputs[0]
        output = self.outputs[0]

        buffer = _input.parent_output._data
        match (self._start, self._length):
            case [None, None]:
                view = buffer
            case [start, None]:
                view = buffer[start:]
            case [None, length]:
                view = buffer[:length]
            case [start, length]:
                view = buffer[start:start+length]

        output._set_data(
            view,
            owns_buffer=False,
            forbid_reallocation=True,
        )
