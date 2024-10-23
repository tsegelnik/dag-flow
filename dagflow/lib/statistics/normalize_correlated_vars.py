from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import add, divide, matmul, multiply, subtract
from scipy.linalg import solve_triangular

from ...core.exception import InitializationError
from ...core.input_handler import MissingInputAddPair
from ...core.node import Node
from ...core.type_functions import (
    check_has_inputs,
    check_input_dimension,
    check_input_matrix_or_diag,
    check_inputs_equivalence,
    check_inputs_multiplicable_mat,
    copy_from_input_to_output,
)

if TYPE_CHECKING:
    from ...core.input import Input


class NormalizeCorrelatedVars(Node):
    """Normalize correlated variables or correlate normal variables with linear expression

    If x is a vector of values, μ are the central values and L is a cholesky decomposition
    of the covariance matrix V=LLᵀ then
    z = L⁻¹(x - μ)
    x = Lz + μ
    """

    __slots__ = ("_mode", "_matrix", "_central")
    _mode: str
    _matrix: Input
    _central: Input

    def __init__(self, *args, mode="forward", **kwargs):
        if mode == "forward":
            mark = "c→u"
        elif mode == "backward":
            mark = "u→c"
        else:
            raise InitializationError(
                f'Invalid NormalizeCorrelatedVars mode={mode}. Expect "forward" or "backward"',
                node=self,
            )

        self._mode = mode

        super().__init__(
            *args,
            missing_input_handler=MissingInputAddPair(),
            **kwargs,
            allowed_kw_inputs=("matrix", "central"),
        )
        self._labels.setdefault("mark", mark)

        self._matrix = self._add_input("matrix", positional=False)
        self._central = self._add_input("central", positional=False)

        self._functions.update(
            {
                "forward_2d": self._fcn_forward_2d,
                "backward_2d": self._fcn_backward_2d,
                "forward_1d": self._fcn_forward_1d,
                "backward_1d": self._fcn_backward_1d,
            }
        )

    def _fcn_forward_2d(self):
        self.inputs.touch()
        L = self._matrix.data
        central = self._central.data
        for _input, _output in zip(self.inputs.iter_data(), self.outputs.iter_data()):
            subtract(_input, central, out=_output)
            solve_triangular(L, _output, lower=True, overwrite_b=True, check_finite=False)

    def _fcn_backward_2d(self):
        self.inputs.touch()
        L = self._matrix.data
        central = self._central.data
        for _input, _output in zip(self.inputs.iter_data(), self.outputs.iter_data()):
            matmul(L, _input, out=_output)
            add(_output, central, out=_output)

    def _fcn_forward_1d(self):
        self.inputs.touch()
        Ldiag = self._matrix.data
        central = self._central.data
        for _input, _output in zip(self.inputs.iter_data(), self.outputs.iter_data()):
            subtract(_input, central, out=_output)
            divide(_output, Ldiag, out=_output)

    def _fcn_backward_1d(self):
        self.inputs.touch()
        Ldiag = self._matrix.data
        central = self._central.data
        for _input, _output in zip(self.inputs.iter_data(), self.outputs.iter_data()):
            multiply(Ldiag, _input, out=_output)
            add(_output, central, out=_output)

    def _typefunc(self) -> None:
        check_has_inputs(self)
        ndim = check_input_matrix_or_diag(self, "matrix", check_square=True)
        check_input_dimension(self, "central", 1)
        check_inputs_equivalence(self, ("central", slice(None)))
        check_inputs_multiplicable_mat(self, "matrix", slice(None))
        copy_from_input_to_output(self, slice(None), slice(None))

        key = f"{self._mode}_{ndim}d"
        try:
            self.fcn = self._functions[key]
        except KeyError as exc:
            raise InitializationError(
                f'Invalid mode "{key}". Expect: {self._functions.keys()}'
            ) from exc
