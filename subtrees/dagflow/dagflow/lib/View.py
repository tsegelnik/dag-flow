from ..nodes import FunctionNode
from ..typefunctions import (
    copy_input_dtype_to_output,
    copy_input_shape_to_output,
)

class View(FunctionNode):
    """Creates a node with a single data output which is a view on the input"""

    def __init__(self, name, outname="view", **kwargs):
        super().__init__(name, **kwargs)
        output = self._add_output(
            outname, allocatable=False, forbid_reallocation=True
        )
        self._add_input("input", child_output=output)

    def _fcn(self, _, inputs, outputs):
        return self.inputs[0].data

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        copy_input_dtype_to_output(self, 0, 0)
        copy_input_shape_to_output(self, 0, 0)

    def _post_allocate(self) -> None:
        input = self.inputs[0]
        output = self.outputs[0]
        output._set_data(
            input.parent_output._data,
            owns_buffer=False,
            forbid_reallocation=True,
        )
