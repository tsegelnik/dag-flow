from typing import TYPE_CHECKING, Literal

from numpy import matmul, multiply

from ..exception import TypeFunctionError
from ..nodes import FunctionNode
from ..typefunctions import (
    check_input_dimension,
    check_input_matrix_or_diag,
    check_inputs_multiplicable_mat,
    eval_output_dtype,
)

if TYPE_CHECKING:
    from ..input import Input
    from ..output import Output


class VectorMatrixProduct(FunctionNode):
    """
    Compute matrix product `C=row(v)@M` or `C=M@column(v)`
    """

    __slots__ = ("_vec", "_mat", "_out", "_matrix_column")

    _vec: "Input"
    _mat: "Input"
    _out: "Output"
    _matrix_column: bool

    def __init__(
        self, *args, mode: Literal["column", "row"] = "column", **kwargs
    ) -> None:
        super().__init__(*args, **kwargs, allowed_kw_inputs=("vec", "mat"))
        self._vec = self._add_input("vector")
        self._mat = self._add_input("matrix")
        self._out = self._add_output("result")

        if mode == "column":
            self._matrix_column = True
            self._labels.setdefault("mark", "M@col(v)")
        elif mode == "row":
            self._matrix_column = False
            self._labels.setdefault("mark", "row(v)@M")
        else:
            raise RuntimeError(f"Invalid VectorMatrixProduct mode {mode}")

        self._functions.update(
            {
                "row_diagonal": self._fcn_row_diagonal,
                "row_block": self._fcn_row_block,
                "diagonal_column": self._fcn_diagonal_column,
                "block_column": self._fcn_block_column,
            }
        )

    def _fcn_row_block(self):
        row = self._vec.data
        mat = self._mat.data
        out = self._out.data
        matmul(row, mat, out=out)

    def _fcn_block_column(self):
        column = self._vec.data
        mat = self._mat.data
        out = self._out.data
        matmul(mat, column, out=out)

    def _fcn_row_diagonal(self):
        row = self._vec.data
        diag = self._mat.data
        out = self._out.data
        multiply(row, diag, out=out)

    def _fcn_diagonal_column(self):
        col = self._vec.data
        diag = self._mat.data
        out = self._out.data
        multiply(diag, col, out=out)

    def _typefunc(self) -> None:
        check_input_dimension(self, "vector", ndim=1)
        ndim_mat = check_input_matrix_or_diag(self, "matrix")
        if ndim_mat not in (1, 2):
            raise TypeFunctionError(f"Matrix dimension >2: {ndim_mat}", node=self)

        if self._matrix_column:
            (resshape,) = check_inputs_multiplicable_mat(self, "matrix", "vector")
            self.fcn = (
                ndim_mat == 2 and self._fcn_block_column or self._fcn_diagonal_column
            )
            self._out.dd.shape = (resshape[0],)
        else:
            (resshape,) = check_inputs_multiplicable_mat(self, "vector", "matrix")
            self.fcn = ndim_mat == 2 and self._fcn_row_block or self._fcn_row_diagonal
            self._out.dd.shape = (resshape[-1],)

        eval_output_dtype(self, slice(None), "result")
