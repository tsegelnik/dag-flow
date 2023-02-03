from ..input_extra import MissingInputAddPair
from ..nodes import FunctionNode
from ..typefunctions import (
    check_has_inputs,
    copy_input_to_output,
    check_input_square
)
from scipy.linalg import cholesky

class Cholesky(FunctionNode):
    """Compute the cholesky decomposition of a matrix V=LL̃ᵀ"""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault(
                "missing_input_handler", MissingInputAddPair(input_fmt='matrix', output_fmt='L')
        )
        super().__init__(*args, **kwargs)

    def _fcn(self, _, inputs, outputs):
        """Compute Cholesky decomposition using `scipy.linalg.cholesky
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

    def _typefunc(self) -> None:
        check_has_inputs(self)
        check_input_square(self, slice(None))
        copy_input_to_output(self, slice(None), slice(None))
