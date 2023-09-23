from typing import TYPE_CHECKING

from numpy import (
    add,
    copyto,
    divide,
    matmul,
    multiply,
    ndarray,
    subtract,
    zeros,
)
from scipy.linalg import solve_triangular

from ..nodes import FunctionNode
from ..typefunctions import (
    check_has_inputs,
    check_input_dimension,
    check_input_matrix_or_diag,
    check_inputs_equivalence,
    check_inputs_multiplicable_mat,
    copy_from_input_to_output,
)

if TYPE_CHECKING:
    from ..input import Input
    from ..output import Output


class NormalizeCorrelatedVars2(FunctionNode):
    """Normalize correlated variables or correlate normal variables with linear expression

    If x is a vector of values, μ are the central values and L is a cholesky decomposition
    of the covariance matrix V=LLᵀ then
    z = L⁻¹(x - μ)
    x = Lz + μ
    """

    __slots__ = (
        "_ndim",
        "_matrix",
        "_central",
        "_input_value",
        "_input_normvalue",
        "_output_value",
        "_output_normvalue",
        "_valuedata",
        "_normvaluedata",
    )

    _ndim: str
    _matrix: "Input"
    _central: "Input"
    _input_value: "Input"
    _input_normvalue: "Input"
    _output_value: "Output"
    _output_normvalue: "Output"
    _valuedata: ndarray
    _normvaluedata: ndarray

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            **kwargs,
            allowed_inputs=(
                "matrix",
                "central",
                "value",
                "normvalue",
            ),
        )
        self._labels.setdefault("mark", "c↔u")

        self._matrix = self._add_input("matrix", positional=False)
        self._central = self._add_input("central", positional=False)

        self._input_value, self._output_value = self._add_pair(
            "value",
            "value",
            input_kws={"allocatable": True},
            output_kws={"forbid_reallocation": True, "allocatable": False},
        )
        self._input_normvalue, self._output_normvalue = self._add_pair(
            "normvalue",
            "normvalue",
            input_kws={"allocatable": True},
            output_kws={"forbid_reallocation": True, "allocatable": False},
        )

        self._functions.update(
            {
                "forward_2d": self._fcn_forward_2d,
                "forward_1d": self._fcn_forward_1d,
                "backward_2d": self._fcn_backward_2d,
                "backward_1d": self._fcn_backward_1d,
            }
        )

    def _fcn_forward_2d(self):
        self.inputs.touch()
        L = self._matrix.data
        central = self._central.data
        input_value = self._input_value.data
        output_value = self._output_value.data
        output_normvalue = self._output_normvalue.data

        subtract(input_value, central, out=output_normvalue)
        solve_triangular(
            L,
            output_normvalue,
            lower=True,
            overwrite_b=True,
            check_finite=False,
        )
        copyto(output_value, input_value)

    def _fcn_backward_2d(self):
        self.inputs.touch()
        L = self._matrix.data
        central = self._central.data
        input_normvalue = self._input_normvalue.data
        output_value = self._output_value.data
        output_normvalue = self._output_normvalue.data

        matmul(L, input_normvalue, out=output_value)
        add(output_value, central, out=output_value)
        copyto(output_normvalue, input_normvalue)

    def _fcn_forward_1d(self):
        self.inputs.touch()
        Ldiag = self._matrix.data
        central = self._central.data
        input_value = self._input_value.data
        output_value = self._output_value.data
        output_normvalue = self._output_normvalue.data

        subtract(input_value, central, out=output_normvalue)
        divide(output_normvalue, Ldiag, out=output_normvalue)
        copyto(output_value, input_value)

    def _fcn_backward_1d(self):
        self.inputs.touch()
        Ldiag = self._matrix.data
        central = self._central.data
        input_normvalue = self._input_normvalue.data
        output_value = self._output_value.data
        output_normvalue = self._output_normvalue.data

        multiply(Ldiag, input_normvalue, out=output_value)
        add(output_value, central, out=output_value)
        copyto(output_normvalue, input_normvalue)

    def _on_taint(self, caller: "Input") -> None:
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
        ndim = check_input_matrix_or_diag(self, "matrix", check_square=True)
        check_input_dimension(self, "central", 1)
        check_inputs_equivalence(self, ("central", slice(None)))
        check_inputs_multiplicable_mat(self, "matrix", slice(None))
        copy_from_input_to_output(self, slice(None), slice(None))

        self.labels.inherit(
            self._input_value.parent_node.labels,
            fmtlong="[norm] {}",
            fmtshort="n({})",
        )

        self._ndim = f"{ndim}d"
        self.fcn = self._functions[f"forward_{self._ndim}"]

        self._valuedata = zeros(
            shape=self._input_value.dd.shape, dtype=self._input_value.dd.dtype
        )
        self._normvaluedata = zeros(
            shape=self._input_normvalue.dd.shape,
            dtype=self._input_normvalue.dd.dtype,
        )
        self._input_value.set_own_data(self._valuedata, owns_buffer=False)
        self._input_normvalue.set_own_data(self._normvaluedata, owns_buffer=False)
        self._output_value._set_data(
            self._valuedata, owns_buffer=False, forbid_reallocation=True
        )
        self._output_normvalue._set_data(
            self._normvaluedata, owns_buffer=False, forbid_reallocation=True
        )
