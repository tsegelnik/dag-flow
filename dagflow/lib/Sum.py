from numpy import copyto, result_type

from ..input_extra import MissingInputAddOne
from ..nodes import FunctionNode

class Sum(FunctionNode):
    """Sum of all the inputs together"""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault(
            "missing_input_handler", MissingInputAddOne(output_fmt="result")
        )
        super().__init__(*args, **kwargs)

    def _fcn(self, _, inputs, outputs):
        out = outputs["result"].data
        copyto(out, inputs[0].data)
        if len(inputs) > 1:
            for input in inputs[1:]:
                out += input.data
        return out

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        self.outputs["result"]._shape = self.inputs[0].shape
        self.outputs["result"]._dtype = result_type(
            *tuple(inp.dtype for inp in self.inputs)
        )
