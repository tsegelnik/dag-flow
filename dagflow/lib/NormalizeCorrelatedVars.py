from ..input_extra import MissingInputAddPair
from ..nodes import FunctionNode
from ..typefunctions import (
    check_has_inputs,
    copy_input_to_output
)
from ..exception import InitializationError

from scipy.linalg import solve_triangular
from numpy import matmul

class NormalizeCorrelatedVars(FunctionNode):
    """Product of all the inputs together"""

    _mode: str
    def __init__(self, *args, mode='cholesky', **kwargs):
        self._mode = mode

        kwargs.setdefault(
            "missing_input_handler", MissingInputAddPair()
        )
        super().__init__(*args, **kwargs)

        self._add_input("matrix", positional=False)
        self._add_input("central", positional=False)

        self._functions.update({
                "cholesky":           self._fcn_cholesky,
                "cholesky_reverse":   self._fcn_cholesky_reverse,
                # "covariance":         self._fcn_covariance
                # "covariance_reverse": self._fcn_covariance_reverse
                })
        try:
            self.fcn = self._functions[self._mode]
        except KeyError:
            raise InitializationError(f'Invalid mode "{self._mode}". Expect: {self._functions.keys()}')

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
