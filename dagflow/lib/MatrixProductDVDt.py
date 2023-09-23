from typing import TYPE_CHECKING

from numpy import empty, matmul, multiply
from numpy.typing import NDArray

from ..nodes import FunctionNode
from ..typefunctions import (
    check_has_inputs,
    check_input_dimension,
    check_input_matrix_or_diag,
    check_inputs_multiplicable_mat,
    eval_output_dtype,
)

if TYPE_CHECKING:
    from ..input import Input
    from ..output import Output


class MatrixProductDVDt(FunctionNode):
    """
    Compute matrix product `C=L@D@Lᵀ`,
    where `L` is a matrix and `D` is a diagonal matrix (maybe 1d array).

    The node refers to the LDLT decomposition.
    """

    __slots__ = ("_left", "_square", "_out", "_buffer")

    _left: "Input"
    _square: "Input"
    _out: "Output"
    _buffer: NDArray

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, allowed_inputs=("left", "square"))
        self._left = self._add_input("left")
        self._square = self._add_input("square")
        self._out = self._add_output("result")
        self._functions.update(
            {"diagonal": self._fcn_diagonal, "square": self._fcn_square}
        )
        self._labels.setdefault("mark", "LDLᵀ")

    def _fcn_diagonal(self):
        left = self._left.data
        diagonal = self._square.data  # square matrix stored as diagonal
        out = self._out.data
        multiply(left, diagonal, out=self._buffer)
        matmul(self._buffer, left.T, out=out)

    def _fcn_square(self):
        left = self._left.data
        square = self._square.data
        out = self._out.data
        matmul(left, square, out=self._buffer)
        matmul(self._buffer, left.T, out=out)

    def _typefunc(self) -> None:
        check_has_inputs(self, ("left", "square"))
        check_input_dimension(self, "left", ndim=2)
        ndim = check_input_matrix_or_diag(self, "square", check_square=True)
        check_inputs_multiplicable_mat(self, "left", "square")
        eval_output_dtype(self, slice(None), "result")
        self._out.dd.shape = (self._left.dd.shape[0], self._left.dd.shape[0])
        self.fcn = self._functions["diagonal" if ndim == 1 else "square"]

    def _post_allocate(self) -> None:
        self._buffer = empty(shape=self._left.dd.shape, dtype=self._left.dd.dtype)
