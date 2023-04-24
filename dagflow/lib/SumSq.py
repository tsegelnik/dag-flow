from numpy import add, square, ndarray, empty_like

from ..input_extra import MissingInputAddOne
from ..nodes import FunctionNode
from ..typefunctions import (
    check_has_inputs,
    eval_output_dtype,
    copy_input_shape_to_output,
    check_inputs_equivalence,
    AllPositionals
)

class SumSq(FunctionNode):
    """Sum of the squared of all the inputs"""

    _buffer: ndarray
    _mark = 'Σ()²'
    def __init__(self, *args, **kwargs):
        kwargs.setdefault(
            "missing_input_handler", MissingInputAddOne(output_fmt="result")
        )
        super().__init__(*args, **kwargs)

    def _fcn(self, _, inputs, outputs):
        out = outputs["result"].data
        square(inputs[0].data, out=out)
        if len(inputs) > 1:
            for input in inputs[1:]:
                square(input.data, out=self._buffer)
                add(self._buffer, out, out=out)
        return out

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        check_has_inputs(self)
        copy_input_shape_to_output(self, 0, "result")
        check_inputs_equivalence(self)
        eval_output_dtype(self, AllPositionals, "result")

    def _post_allocate(self) -> None:
        self._buffer = empty_like(self.inputs[0].get_data_unsafe())
