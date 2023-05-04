from numpy import copyto, empty_like, multiply, matmul, ndarray
from numpy.typing import NDArray

from ..nodes import FunctionNode
from ..input import Input
from ..output import Output
# from ..input_extra import MissingInputAddOne
from ..typefunctions import (
    check_has_inputs,
    eval_output_dtype,
    check_input_dimension,
    check_input_square_or_diag,
    check_inputs_multiplicable_mat
)


class MatrixProductDVDt(FunctionNode):
    _left: Input
    _square: Input
    _out: Output
    _buffer: ndarray

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._left = self.add_input("left")
        self._square = self.add_input("square")
        self._out = self.add_output("result")
        self._functions.update({
            "diagonal": self._fcn_diagonal,
            "square": self._fcn_square
            })

    def _fcn_diagonal(self, inputs, outputs):
        left = self._left.data
        diagonal = self._square.data    # square matrix stored as diagonal
        out = self._out.data
        multiply(left, diagonal, out=self._buffer)
        matmul(self._buffer, left.T, out=out)
        return out

    def _fcn_square(self, inputs, outputs):
        out = outputs["result"].data
        left = inputs["left"].data
        square = inputs["square"].data
        matmul(left, square, out=self._buffer)
        matmul(self._buffer, left.T, out=out)
        return out

    def _typefunc(self) -> None:
        check_has_inputs(self, ["left", "square"])
        check_input_dimension(self, "left", ndim=2)
        ndim = check_input_square_or_diag(self, "square")
        check_inputs_multiplicable_mat(self, "left", "square")
        eval_output_dtype(self, slice(None), "result")
        self._out.dd.shape=((self._left.dd.shape[0], 
                             self._left.dd.shape[0]))
        if ndim == 1:
            self._fcn = self._functions["diagonal"]
        else:
            self._fcn = self._functions["square"]

    def _post_allocate(self):
        self._buffer = empty_like(self.inputs["left"].get_data_unsafe())
