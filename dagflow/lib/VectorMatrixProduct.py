from typing import TYPE_CHECKING

from numpy import matmul, multiply
from numpy.typing import NDArray

from ..nodes import FunctionNode
from ..typefunctions import (
    check_has_inputs,
    check_input_dimension,
    check_input_matrix_or_diag,
    check_inputs_multiplicable_mat,
    eval_output_dtype,
)
from ..exception import TypeFunctionError

if TYPE_CHECKING:
    from ..input import Input
    from ..output import Output


class VectorMatrixProduct(FunctionNode):
    """
    Compute matrix product `C=row(A)@B`,
    """

    __slots__ = ("_vec", "_mat", "_out")

    _vec: "Input"
    _mat: "Input"
    _out: "Output"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._vec = self._add_input("vector")
        self._mat = self._add_input("matrix")
        self._out = self._add_output("result")
        self._labels.setdefault("mark", "row(A)@B")

        self._functions.update(
            {
                "diagonal": self._fcn_diagonal,
                "block":    self._fcn_block,
            }
        )

    def _fcn_block(self) -> NDArray:
        row = self._vec.data
        mat = self._mat.data
        out = self._out.data
        matmul(row, mat, out=out)

    def _fcn_diagonal(self) -> NDArray:
        row = self._vec.data
        diag = self._mat.data
        out = self._out.data
        multiply(row, diag, out=out)

    def _typefunc(self) -> None:
        check_has_inputs(self, ("vector", "matrix"))
        check_input_dimension(self, "vector", ndim=1)
        ndim_mat = check_input_matrix_or_diag(self, "matrix")
        if ndim_mat==2:
            self.fcn = self._fcn_block
        elif ndim_mat==1:
            self.fcn = self._fcn_diagonal
        else:
            raise TypeFunctionError(f"Matrix dimension >2: {ndim_mat}", node=self)

        resshape, = check_inputs_multiplicable_mat(self, "vector", "matrix")
        eval_output_dtype(self, slice(None), "result")

        self._out.dd.shape = resshape[:ndim_mat]

