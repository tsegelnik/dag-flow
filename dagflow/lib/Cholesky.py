from numpy import sqrt
from scipy.linalg import cholesky

from ..inputhandler import MissingInputAddPair
from ..nodes import FunctionNode
from ..typefunctions import check_has_inputs
from ..typefunctions import check_input_matrix_or_diag
from ..typefunctions import copy_from_input_to_output


class Cholesky(FunctionNode):
    """Compute the Cholesky decomposition of a matrix V=LL̃ᵀ
    1d input is considered to be a diagonal of square matrix"""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault(
            "missing_input_handler",
            MissingInputAddPair(input_fmt="matrix", output_fmt="L"),
        )
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "V→L")

        self._functions.update({"square": self._fcn_square, "diagonal": self._fcn_diagonal})

    def _fcn_square(self):
        """Compute Cholesky decomposition using `scipy.linalg.cholesky`
        NOTE: inplace computation (`overwrite_a=True`) works only for
        the F-based arrays. As soon as by default C-arrays are used,
        transposition produces an F-array (view). Transposition with
        `lower=False` produces a lower matrix in the end.
        """
        self.inputs.touch()

        for _input, _output in zip(self.inputs.iter_data(), self.outputs.iter_data()):
            _output[:] = _input
            cholesky(_output.T, overwrite_a=True, lower=False)  # produces L (!) inplace

    def _fcn_diagonal(self):
        """Compute "Cholesky" decomposition using of a diagonal of a square matrix.
        Elementwise sqrt is used.
        """
        self.inputs.touch()

        for _input, _output in zip(self.inputs.iter_data(), self.outputs.iter_data()):
            sqrt(_input, out=_output)

    def _typefunc(self) -> None:
        check_has_inputs(self)
        ndim = check_input_matrix_or_diag(self, slice(None), check_square=True)
        copy_from_input_to_output(self, slice(None), slice(None))

        if ndim == 2:
            self.fcn = self._functions["square"]
            self._mark = "V→L"
        else:
            self.fcn = self._functions["diagonal"]
            self._mark = "sqrt(Vᵢ)"
