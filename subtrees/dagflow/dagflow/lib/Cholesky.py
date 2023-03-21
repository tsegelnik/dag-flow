from ..input_extra import MissingInputAddPair
from ..nodes import FunctionNode
from ..typefunctions import (
    check_has_inputs,
    copy_input_to_output,
    check_input_square_or_diag
)
from scipy.linalg import cholesky
from numpy import sqrt

class Cholesky(FunctionNode):
    """Compute the Cholesky decomposition of a matrix V=LL̃ᵀ
    1d input is considered to be a diagonal of square matrix"""
    _mark: str = 'V→L'
    def __init__(self, *args, **kwargs):
        kwargs.setdefault(
                "missing_input_handler", MissingInputAddPair(input_fmt='matrix', output_fmt='L')
        )
        super().__init__(*args, **kwargs)

        self._functions.update({
                "square": self._fcn_square,
                "diagonal": self._fcn_diagonal
            })

    def _fcn_square(self, _, inputs, outputs):
        """Compute Cholesky decomposition using `scipy.linalg.cholesky`
        NOTE: inplace computation (`overwrite_a=True`) works only for
        the F-based arrays. As soon as by default C-arrays are used,
        transposition produces an F-array (view). Transposition with
        `lower=False` produces a lower matrix in the end.
        """
        inputs.touch()

        for input, output in zip(inputs.iter_data(), outputs.iter_data()):
            output[:] = input
            cholesky(output.T, overwrite_a=True, lower=False) # produces L (!) inplace
            # output[:]=cholesky(input, lower=True)

    def _fcn_diagonal(self, _, inputs, outputs):
        """Compute "Cholesky" decomposition using of a diagonal of a square matrix.
        Elementwise sqrt is used.
        """
        inputs.touch()

        for input, output in zip(inputs.iter_data(), outputs.iter_data()):
            sqrt(input, out=output)

    def _typefunc(self) -> None:
        check_has_inputs(self)
        ndim = check_input_square_or_diag(self, slice(None))
        copy_input_to_output(self, slice(None), slice(None))

        if ndim==2:
            self.fcn = self._functions["square"]
            self._mark = 'V→L'
        else:
            self.fcn = self._functions["diagonal"]
            self._mark = 'sqrt(Vᵢ)'

