from numpy import diag, log, empty
from numpy.typing import NDArray

from ..inputhandler import MissingInputAddPair
from ..node import Node
from ..typefunctions import check_has_inputs, check_input_matrix_or_diag, copy_input_dtype_to_output, AllPositionals


class LogProdDiag(Node):
    """
    Compute the LogProdDiag of a matrix log|V|=log|LL̃ᵀ|->Σlog(Lᵢᵢ)
    based on Cholesky decomposition of matrix V
    1d input is considered to be a squared root diagonal of square matrix
    
    inputs:
        `matrix`: cholesky decomposition of matrix

    outputs:
        `0`: sum of logarithm of diagonal elements

    .. note:: factor 2 is skipped
    """

    __slots__ = ("_buffer",)
    _buffer: NDArray

    def __init__(self, *args, **kwargs):
        kwargs.setdefault(
            "missing_input_handler",
            MissingInputAddPair(input_fmt="matrix", output_fmt="Σlog(Lᵢᵢ)"),
        )
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "log|V|→Σlog(Lᵢᵢ)")

        self._functions.update({"square": self._fcn_square, "diagonal": self._fcn_diagonal})

    def _fcn_square(self):
        """Compute logarithm of determinant of matrix using Cholesky decomposition
        """
        self.inputs.touch()

        for inp, out in zip(self.inputs, self.outputs):
            log(diag(inp.data), out=self._buffer)
            out.data[0] = self._buffer.sum()

    def _fcn_diagonal(self):
        """Compute "LogProdDiag" using of a diagonal of a square matrix.
        """
        self.inputs.touch()

        for inp, out in zip(self.inputs, self.outputs):
            log(inp.data, out=self._buffer)
            out.data[0] = self._buffer.sum()

    def _typefunc(self) -> None:
        check_has_inputs(self, AllPositionals)
        ndim = check_input_matrix_or_diag(self, AllPositionals, check_square=True)
        copy_input_dtype_to_output(self, AllPositionals, AllPositionals)
        for out in self.outputs:
            out.dd.shape = (1,)

        if ndim == 2:
            self.fcn = self._functions["square"]
            self.labels.mark = "Σlog(Lᵢᵢ)"
        else:
            self.fcn = self._functions["diagonal"]
            self.labels.mark = "Σlog(Dᵢᵢ)"

    def _post_allocate(self) -> None:
        inpdd = self.inputs[0].dd
        self._buffer = empty(shape=(inpdd.shape[0],), dtype=inpdd.dtype)
