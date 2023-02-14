from ..input_extra import MissingInputAddPair
from ..nodes import FunctionNode
from ..node import Input, Output
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
from numpy import matmul, subtract, divide, multiply, add, zeros

class NormalizeCorrelatedVars2(FunctionNode):
    """Normalize correlated variables or correlate normal variables with linear expression

    If x is a vector of values, μ are the central values and L is a cholesky decomposition
    of the covariance matrix V=LLᵀ then
    z = L⁻¹(x - μ)
    x = Lz + μ
    """

    _mark: str = 'c↔u'

    _input_value: Input
    _input_normvalue: Input
    _output_value: Output
    _output_normvalue: Output

    _ndim: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._add_input("matrix", positional=False)
        self._add_input("central", positional=False)

        self._input_value, self._output_value = self._add_pair(
            "value", "value",
            input_kws={'allocatable': True},
            output_kws={'forbid_reallocation': True, 'allocatable': False},
        )
        self._input_normvalue, self._output_normvalue = self._add_pair(
            "normvalue", "normvalue",
            input_kws={'allocatable': True},
            output_kws={'forbid_reallocation': True, 'allocatable': False},
        )

        self._functions.update({
                "forward_2d":  self._fcn_forward_2d,
                "forward_1d":  self._fcn_forward_1d,
                "backward_2d":  self._fcn_backward_2d,
                "backward_1d":  self._fcn_backward_1d,
                })

    def _fcn_forward_2d(self, _, inputs, outputs):
        inputs.touch()
        L = inputs["matrix"].data
        central = inputs["central"].data

        input = inputs["value"].data
        output = outputs["normvalue"].data

        subtract(input, central, out=output)
        solve_triangular(L, output, lower=True, overwrite_b=True, check_finite=False)

    def _fcn_backward_2d(self, _, inputs, outputs):
        inputs.touch()
        L = inputs["matrix"].data
        central = inputs["central"].data

        input = inputs["normvalue"].data
        output = outputs["value"].data

        matmul(L, input, out=output)
        add(output, central, out=output)

    def _fcn_forward_1d(self, _, inputs, outputs):
        inputs.touch()
        Ldiag = inputs["matrix"].data
        central = inputs["central"].data

        input = inputs["value"].data
        output = outputs["normvalue"].data

        subtract(input, central, out=output)
        divide(output, Ldiag, out=output)

    def _fcn_backward_1d(self, _, inputs, outputs):
        inputs.touch()
        Ldiag = inputs["matrix"].data
        central = inputs["central"].data

        input = inputs["normvalue"].data
        output = outputs["value"].data

        multiply(Ldiag, input, out=output)
        add(output, central, out=output)

    def _on_taint(self, caller: Input) -> None:
        if caller is self._input_value:
            self.fcn = self._functions[f"forward_{self._ndim}"]
        else:
            self.fcn = self._functions[f"backward_{self._ndim}"]

    def _typefunc(self) -> None:
        check_has_inputs(self)
        ndim = check_input_square_or_diag(self, 'matrix')
        check_input_dimension(self, 'central', 1)
        check_inputs_equivalence(self, ('central', slice(None)))
        check_inputs_multiplicable_mat(self, 'matrix', slice(None))
        copy_input_to_output(self, slice(None), slice(None))

        self._ndim=f"{ndim}d"
        self.fcn = self._functions[f"forward_{self._ndim}"]

        self._valuedata = zeros(shape=self._input_value.shape, dtype=self._input_value.dtype)
        self._normvaluedata = zeros(shape=self._input_normvalue.shape, dtype=self._input_normvalue.dtype)
        self._input_value.set_own_data(self._valuedata, owns_buffer=False)
        self._input_normvalue.set_own_data(self._normvaluedata, owns_buffer=False)
        self._output_value._set_data(self._valuedata, owns_buffer=False, forbid_reallocation=True)
        self._output_normvalue._set_data(self._normvaluedata, owns_buffer=False, forbid_reallocation=True)
