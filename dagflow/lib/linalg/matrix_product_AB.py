from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import matmul, multiply

from ...exception import TypeFunctionError
from ...node import Node
from ...type_functions import (
    check_has_inputs,
    check_input_matrix_or_diag,
    check_inputs_multiplicable_mat,
    eval_output_dtype,
)

if TYPE_CHECKING:
    from ...input import Input
    from ...output import Output


class MatrixProductAB(Node):
    """
    Compute matrix product `C=A@B`,
    """

    __slots__ = ("_left", "_right", "_out")

    _left: Input
    _right: Input
    _out: Output

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, allowed_kw_inputs=("left", "right"))
        self._left = self._add_input("left")
        self._right = self._add_input("right")
        self._out = self._add_output("result")
        self._labels.setdefault("mark", "A@B")

        self._functions.update(
            {
                "diagonal_diagonal": self._fcn_diagonal_diagonal,
                "diagonal_block": self._fcn_diagonal_block,
                "block_diagonal": self._fcn_block_diagonal,
                "block_block": self._fcn_block_block,
            }
        )

    def _fcn_block_block(self):
        left = self._left.data
        right = self._right.data
        out = self._out.data
        matmul(left, right, out=out)

    def _fcn_block_diagonal(self):
        left = self._left.data
        right = self._right.data
        out = self._out.data
        multiply(left, right, out=out)

    def _fcn_diagonal_block(self):
        left = self._left.data
        right = self._right.data
        out = self._out.data
        multiply(left[:, None], right, out=out)

    def _fcn_diagonal_diagonal(self):
        left = self._left.data
        right = self._right.data
        out = self._out.data
        multiply(left, right, out=out)

    def _typefunc(self) -> None:
        check_has_inputs(self, ("left", "right"))
        ndim_left = check_input_matrix_or_diag(self, "left")
        ndim_right = check_input_matrix_or_diag(self, "right")

        ndim = ndim_left, ndim_right
        ndim_out = 2
        if ndim == (2, 2):
            self.fcn = self._fcn_block_block
        elif ndim == (2, 1):
            self.fcn = self._fcn_block_diagonal
        elif ndim == (1, 2):
            self.fcn = self._fcn_diagonal_block
        elif ndim == (1, 1):
            self.fcn = self._fcn_diagonal_diagonal
            ndim_out = 1
        else:
            raise TypeFunctionError(f"One of the inputs' dimension >2: {ndim}", node=self)

        (resshape,) = check_inputs_multiplicable_mat(self, "left", "right")
        eval_output_dtype(self, slice(None), "result")

        self._out.dd.shape = resshape[:ndim_out]
