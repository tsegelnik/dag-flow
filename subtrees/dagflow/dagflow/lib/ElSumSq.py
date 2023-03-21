from numpy import ndarray
from numpy.typing import NDArray

from numba import njit
from ..input_extra import MissingInputAddOne
from ..nodes import FunctionNode
from ..typefunctions import (
    check_has_inputs,
    eval_output_dtype,
    check_inputs_same_dtype,
    AllPositionals
)

@njit(cache=True)
def _sumsq(data: NDArray, out: NDArray):
    sm = 0.0
    for v in data:
        sm+=v*v
    out[0]+=sm

class ElSumSq(FunctionNode):
    """Sum of the squared of all the inputs"""

    _buffer: ndarray
    def __init__(self, *args, **kwargs):
        kwargs.setdefault(
            "missing_input_handler", MissingInputAddOne(output_fmt="result")
        )
        super().__init__(*args, **kwargs)

    def _fcn(self, _, inputs, outputs):
        out = outputs["result"].data
        out[0] = 0.0
        for input in inputs:
            _sumsq(input.data, out)
        return out

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        check_has_inputs(self)
        check_inputs_same_dtype(self)
        eval_output_dtype(self, AllPositionals, "result")
        self.outputs[0].dd.shape=(1,)
