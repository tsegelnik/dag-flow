from ..input_extra import MissingInputAddOne
from ..nodes import FunctionNode
from ..typefunctions import (
    check_has_inputs,
    combine_inputs_shape_to_output,
    eval_output_dtype,
)


class Concatenation(FunctionNode):
    """Creates a node with a single data output from all the inputs data"""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault(
            "missing_input_handler", MissingInputAddOne(output_fmt="result")
        )
        super().__init__(*args, **kwargs)

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        check_has_inputs(self)
        combine_inputs_shape_to_output(self, slice(None), "result")
        eval_output_dtype(self, slice(None), "result")

    def _fcn(self, _, inputs, outputs):
        res = outputs["result"].data
        res[:] = (inp.data for inp in inputs)
        return res
