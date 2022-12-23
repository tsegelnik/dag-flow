from numpy import result_type

from ..input_extra import MissingInputAddOne
from ..nodes import FunctionNode

class Concatenation(FunctionNode):
    """Creates a node with a single data output from all the inputs data"""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault(
            "missing_input_handler", MissingInputAddOne(output_fmt="result")
        )
        super().__init__(*args, **kwargs)

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        self.outputs["result"]._shape = tuple(inp.shape for inp in self.inputs)
        self.outputs["result"]._dtype = result_type(
            *tuple(inp.dtype for inp in self.inputs)
        )

    def _fcn(self, _, inputs, outputs):
        res = outputs["result"].data
        res[:] = (inp.data for inp in inputs)
        return res
