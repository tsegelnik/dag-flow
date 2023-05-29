from typing import TYPE_CHECKING, Optional

from ..nodes import FunctionNode
from ..typefunctions import copy_from_input_to_output
if TYPE_CHECKING:
    from ..output import Output
    from ..input import Input

class View(FunctionNode):
    """Creates a node with a single data output which is a view on the input"""
    __slots__ = ('_input',)
    _input: "Input"

    def __init__(self, name, output: Optional["Output"]=None, *, outname="view", **kwargs):
        super().__init__(name, **kwargs)
        child_output = self._add_output(
            outname, allocatable=False, forbid_reallocation=True
        )
        self._input = self._add_input("input", child_output=child_output)

        if output is not None:
            output >> self._input
            if output.closed:
                self.close()

    def _fcn(self, _, inputs, outputs):
        return self._input.data

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        copy_from_input_to_output(self, 0, 0)

    def _post_allocate(self) -> None:
        input = self.inputs[0]
        output = self.outputs[0]
        output._set_data(
            input.parent_output._data,
            owns_buffer=False,
            forbid_reallocation=True,
        )
