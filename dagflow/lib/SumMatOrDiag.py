from numpy import copyto, add
from numpy.typing import NDArray
from numba import njit

from ..input_extra import MissingInputAddOne
from ..nodes import FunctionNode
from ..typefunctions import (
    check_has_inputs,
    eval_output_dtype,
    copy_input_shape_to_output,
    check_inputs_square_or_diag,
    check_inputs_same_dtype,
    AllPositionals
)

@njit(cache=True)
def _settodiag1(inarray: NDArray, outmatrix: NDArray):
    for i in range(inarray.size):
        outmatrix[i, i] = inarray[i]

@njit(cache=True)
def _addtodiag(inarray: NDArray, outmatrix: NDArray):
    for i in range(inarray.size):
        outmatrix[i, i] += inarray[i]

class SumMatOrDiag(FunctionNode):
    """Sum of all the inputs together. Inputs are square matrices or diagonals of square matrices"""

    _ndim: int = 0
    def __init__(self, *args, **kwargs):
        kwargs.setdefault(
            "missing_input_handler", MissingInputAddOne(output_fmt="result")
        )
        super().__init__(*args, **kwargs)

        self._functions.update({
                "2d":  self._fcn2d,
                "1d":  self._fcn1d,
                })

    def _fcn2d(self, _, inputs, outputs):
        out = outputs["result"].data
        inp = inputs[0].data
        if len(inp.shape)==1:
            _settodiag1(inp, out)
        else:
            out[:] = inp
        if len(inputs) > 1:
            for input in inputs[1:]:
                if len(input.dd.shape)==1:
                    _addtodiag(input.data, out)
                else:
                    add(input.data, out, out=out)
        return out

    def _fcn1d(self, _, inputs, outputs):
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
        self._ndim = check_inputs_square_or_diag(self)
        check_inputs_same_dtype(self)
        eval_output_dtype(self, AllPositionals, "result")

        size = self.inputs[0].dd.shape[0]
        output = self.outputs[0]
        if self._ndim==2:
            output.dd.shape = size, size
        elif self._ndim==1:
            output.dd.shape = size,
        else:
            assert False

        self.fcn = self._functions[f"{self._ndim}d"]
