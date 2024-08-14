from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import matmul

from ..node import Node
from ..typefunctions import check_input_dimension, eval_output_dtype

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..input import Input
    from ..output import Output


class MatrixProductDDt(Node):
    """
    Compute matrix product `C=D@Dᵀ`.
    """

    __slots__ = ("_matrix", "_out", "_buffer")

    _matrix: Input
    _out: Output
    _buffer: NDArray

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, allowed_kw_inputs=("matrix",))
        self._matrix = self._add_input("matrix")
        self._out = self._add_output("result")
        self._labels.setdefault("mark", "DDᵀ")

    def _fcn(self):
        matrix = self._matrix.data
        out = self._out.data
        matmul(matrix, matrix.T, out=out)

    def _typefunc(self) -> None:
        check_input_dimension(self, "matrix", ndim=2)
        eval_output_dtype(self, slice(None), "result")
        self._out.dd.shape = (self._matrix.dd.shape[0], self._matrix.dd.shape[0])
