from __future__ import annotations

from typing import TYPE_CHECKING

from numba import njit

from ...core.type_functions import AllPositionals, check_input_size, check_inputs_same_dtype
from ..abstract import OneToOneNode

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ...core.input import Input


class LinearFunction(OneToOneNode):
    """Calculates y_i = a*x_i + b"""

    __slots__ = ("_a", "_b")
    _a: Input
    _b: Input

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._a = self._add_input("a", positional=False)
        self._b = self._add_input("b", positional=False)

    def _typefunc(self) -> None:
        super()._typefunc()
        check_input_size(self, ("a", "b"), exact=1)
        check_inputs_same_dtype(self, ("a", "b", AllPositionals))

    def _fcn(self):
        a = self._a.data[0]
        b = self._b.data[0]
        for inp, out in zip(self.inputs.iter_data(), self.outputs.iter_data()):
            _linear_function(inp, out, a, b)


@njit(cache=True)
def _linear_function(inp: NDArray, out: NDArray, a: float, b: float):
    for i in range(len(inp)):
        out[i] = a * inp[i] + b
