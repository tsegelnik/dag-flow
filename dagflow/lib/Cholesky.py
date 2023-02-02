from ..input_extra import MissingInputAddPair
from ..nodes import FunctionNode
from ..typefunctions import (
    check_has_inputs,
    copy_input_to_output
)
from scipy.linalg import cholesky
from numpy import matmul

class Cholesky(FunctionNode):
    """Compute the cholesky decomposition of a matrix V=LL̃ᵀ"""

    _mode: str
    def __init__(self, *args, mode='cholesky', **kwargs):
        self._mode = mode

        kwargs.setdefault(
                "missing_input_handler", MissingInputAddPair(inputfmt='matrix', outputfmt='L')
        )
        super().__init__(*args, **kwargs)

    def _fcn_cholesky(self, _, inputs, outputs):
        inputs.touch()

        L = inputs["matrix"].data
        central = inputs["central"].data

        for input, output in zip(inputs.iter_data(), outputs.iter_data()):
            output[:] = input
            output -= central
            solve_triangular(L, output, lower=True, overwrite_b=True)

    # def _fcn_covariance(self, _, inputs, outputs):
    #     inputs.touch()
    #
    #     L = cholesky(inputs["matrix"].data, lower=True)
    #     central = inputs["central"].data
    #
    #     for input, output in zip(inputs.iter_data(), outputs.iter_data()):
    #         output[:] = input
    #         output -= central
    #         solve_triangular(L, output, lower=True, overwrite_b=True)

    def _fcn_cholesky_reverse(self, _, inputs, outputs):
        inputs.touch()

        L = inputs["matrix"].data
        central = inputs["central"].data

        for input, output in zip(inputs.iter_data(), outputs.iter_data()):
            matmul(L, input, out=output)
            output += central

    # def _fcn_covariance_reverse(self, _, inputs, outputs):
    #     inputs.touch()
    #
    #     L = cholesky(inputs["matrix"].data, lower=True)
    #     central = inputs["central"].data
    #
    #     for input, output in zip(inputs.iter_data(), outputs.iter_data()):
    #         matmul(L, input, out=output)
    #         output += central

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        check_has_inputs(self)
        copy_input_to_output(self, slice(None), slice(None))
