from numba import njit
from numpy import add, copyto
from numpy.typing import NDArray

from ...input_handler import MissingInputAddOne
from ...node import Node
from ...type_functions import (
    AllPositionals,
    check_has_inputs,
    check_inputs_consistent_square_or_diag,
    check_inputs_same_dtype,
    copy_input_shape_to_outputs,
    eval_output_dtype,
)


@njit(cache=True)
def _settodiag1(inarray: NDArray, outmatrix: NDArray):
    outmatrix[:] = 0
    for i in range(inarray.size):
        outmatrix[i, i] = inarray[i]


@njit(cache=True)
def _addtodiag(inarray: NDArray, outmatrix: NDArray):
    for i in range(inarray.size):
        outmatrix[i, i] += inarray[i]


class SumMatOrDiag(Node):
    """Sum of all the inputs together. Inputs are square matrices or diagonals of square matrices"""

    __slots__ = ("_ndim",)
    _ndim: int

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("missing_input_handler", MissingInputAddOne(output_fmt="result"))
        super().__init__(*args, **kwargs)
        self._functions.update({2: self._fcn2d, 1: self._fcn1d})

    def _fcn2d(self):
        out = self.outputs["result"].data
        inp = self.inputs[0].data
        if len(inp.shape) == 1:
            _settodiag1(inp, out)
        else:
            out[:] = inp
        if len(self.inputs) > 1:
            for _input in self.inputs[1:]:
                if len(_input.dd.shape) == 1:
                    _addtodiag(_input.data, out)
                else:
                    add(_input.data, out, out=out)

    def _fcn1d(self):
        out = self.outputs["result"].data
        copyto(out, self.inputs[0].data)
        if len(self.inputs) > 1:
            for _input in self.inputs[1:]:
                add(out, _input.data, out=out)

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        check_has_inputs(self)
        copy_input_shape_to_outputs(self, 0, "result")
        self._ndim = check_inputs_consistent_square_or_diag(self)
        check_inputs_same_dtype(self)
        eval_output_dtype(self, AllPositionals, "result")

        size = self.inputs[0].dd.shape[0]
        output = self.outputs[0]
        if self._ndim == 2:
            output.dd.shape = size, size
        elif self._ndim == 1:
            output.dd.shape = (size,)
        else:
            assert False

        self.fcn = self._functions[self._ndim]
