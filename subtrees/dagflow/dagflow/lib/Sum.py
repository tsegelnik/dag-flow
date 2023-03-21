from numpy import copyto, add

from ..input_extra import MissingInputAddOne
from ..nodes import FunctionNode
from ..typefunctions import (
    check_has_inputs,
    eval_output_dtype,
    copy_input_shape_to_output,
    check_inputs_equivalence,
    AllPositionals
)

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
                add(out, input.data, out=out)
        return out

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        check_has_inputs(self)
        copy_input_shape_to_output(self, 0, "result")
        check_inputs_equivalence(self)
        eval_output_dtype(self, AllPositionals, "result")
