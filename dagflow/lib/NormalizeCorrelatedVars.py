from ..input_extra import MissingInputAddPair
from ..nodes import FunctionNode
from ..typefunctions import (
    check_has_inputs,
    check_input_square_or_diag,
    copy_input_to_output,
    check_input_dimension,
    check_inputs_equivalence,
    check_inputs_multiplicable_mat
)
from ..exception import InitializationError

from scipy.linalg import solve_triangular
from numpy import matmul, subtract, divide, multiply, add

class NormalizeCorrelatedVars(FunctionNode):
    """Product of all the inputs together"""

    _mode: str
    def __init__(self, *args, mode='forward', **kwargs):
        self._mode = mode

        kwargs.setdefault(
            "missing_input_handler", MissingInputAddPair()
        )
        super().__init__(*args, **kwargs)

        self._add_input("matrix", positional=False)
        self._add_input("central", positional=False)

        self._functions.update({
                "forward_2d":  self._fcn_forward_2d,
                "backward_2d": self._fcn_backward_2d,
                "forward_1d":  self._fcn_forward_1d,
                "backward_1d": self._fcn_backward_1d
                })

    def _fcn_forward_2d(self, _, inputs, outputs):
        inputs.touch()
        L = inputs["matrix"].data
        central = inputs["central"].data
        for input, output in zip(inputs.iter_data(), outputs.iter_data()):
            subtract(input, central, out=output)
            solve_triangular(L, output, lower=True, overwrite_b=True, check_finite=False)

    def _fcn_backward_2d(self, _, inputs, outputs):
        inputs.touch()
        L = inputs["matrix"].data
        central = inputs["central"].data
        for input, output in zip(inputs.iter_data(), outputs.iter_data()):
            matmul(L, input, out=output)
            add(output, central, out=output)

    def _fcn_forward_1d(self, _, inputs, outputs):
        inputs.touch()
        Ldiag = inputs["matrix"].data
        central = inputs["central"].data
        for input, output in zip(inputs.iter_data(), outputs.iter_data()):
            subtract(input, central, out=output)
            divide(output, Ldiag, out=output)

    def _fcn_backward_1d(self, _, inputs, outputs):
        inputs.touch()
        Ldiag = inputs["matrix"].data
        central = inputs["central"].data
        for input, output in zip(inputs.iter_data(), outputs.iter_data()):
            multiply(Ldiag, input, out=output)
            add(output, central, out=output)

    def _typefunc(self) -> None:
        check_has_inputs(self)
        ndim = check_input_square_or_diag(self, 'matrix')
        check_input_dimension(self, 'central', 1)
        check_inputs_equivalence(self, ('central', slice(None)))
        check_inputs_multiplicable_mat(self, 'matrix', slice(None))
        copy_input_to_output(self, slice(None), slice(None))

        key = f"{self._mode}_{ndim}d"
        try:
            self.fcn = self._functions[key]
        except KeyError:
            raise InitializationError(f'Invalid mode "{key}". Expect: {self._functions.keys()}')
