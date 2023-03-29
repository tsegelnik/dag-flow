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

from scipy.linalg import solve_triangular
from numpy import matmul, subtract, divide, multiply, add, zeros, copyto

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

        input_value = inputs["value"].data
        output_value = outputs["value"].data
        output_normvalue = outputs["normvalue"].data

        subtract(input_value, central, out=output_normvalue)
        solve_triangular(L, output_normvalue, lower=True, overwrite_b=True, check_finite=False)
        copyto(output_value, input_value)

    def _fcn_backward_2d(self, _, inputs, outputs):
        inputs.touch()
        L = inputs["matrix"].data
        central = inputs["central"].data

        input_normvalue = inputs["normvalue"].data
        output_normvalue = outputs["normvalue"].data
        output_value = outputs["value"].data

        matmul(L, input_normvalue, out=output_value)
        add(output_value, central, out=output_value)
        copyto(output_normvalue, input_normvalue)

    def _fcn_forward_1d(self, _, inputs, outputs):
        inputs.touch()
        Ldiag = inputs["matrix"].data
        central = inputs["central"].data

        input_value = inputs["value"].data
        output_value = outputs["value"].data
        output_normvalue = outputs["normvalue"].data

        subtract(input_value, central, out=output_normvalue)
        divide(output_normvalue, Ldiag, out=output_normvalue)
        copyto(output_value, input_value)

    def _fcn_backward_1d(self, _, inputs, outputs):
        inputs.touch()
        Ldiag = inputs["matrix"].data
        central = inputs["central"].data

        input_normvalue = inputs["normvalue"].data
        output_normvalue = outputs["normvalue"].data
        output_value = outputs["value"].data

        multiply(Ldiag, input_normvalue, out=output_value)
        add(output_value, central, out=output_value)
        copyto(output_normvalue, input_normvalue)

    def _on_taint(self, caller: Input) -> None:
        """Choose the function to call based on the modified input:
            - if normvalue is modified, the value should be updated
            - if value is modified, the normvalue should be updated
            - if sigma or central is modified, the normvalue should be updated

            TODO:
                - implement partial taintflag propagation
                - value should not be tainted on sigma/central modificantion
        """
        if caller is self._input_normvalue:
            self.fcn = self._functions[f"backward_{self._ndim}"]
        else:
            self.fcn = self._functions[f"forward_{self._ndim}"]

    def _typefunc(self) -> None:
        check_has_inputs(self)
        ndim = check_input_square_or_diag(self, 'matrix')
        check_input_dimension(self, 'central', 1)
        check_inputs_equivalence(self, ('central', slice(None)))
        check_inputs_multiplicable_mat(self, 'matrix', slice(None))
        copy_input_to_output(self, slice(None), slice(None))

        for k, v in self.inputs['value'].parent_node.labels:
            if k in ('key',):
                continue
            self._label[k] = f'Normalized {v}'

        self._ndim=f"{ndim}d"
        self.fcn = self._functions[f"forward_{self._ndim}"]

        self._valuedata = zeros(shape=self._input_value.dd.shape, dtype=self._input_value.dd.dtype)
        self._normvaluedata = zeros(shape=self._input_normvalue.dd.shape, dtype=self._input_normvalue.dd.dtype)
        self._input_value.set_own_data(self._valuedata, owns_buffer=False)
        self._input_normvalue.set_own_data(self._normvaluedata, owns_buffer=False)
        self._output_value._set_data(self._valuedata, owns_buffer=False, forbid_reallocation=True)
        self._output_normvalue._set_data(self._normvaluedata, owns_buffer=False, forbid_reallocation=True)
