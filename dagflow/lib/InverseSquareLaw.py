from numba import njit
from numpy import pi
from numpy.typing import NDArray

from .OneToOneNode import OneToOneNode


@njit(cache=True)
def _inv_sq_law(data: NDArray, out: NDArray):
    out[:] = 0.5 / pi / (data[:] * data[:])


class InverseSquareLaw(OneToOneNode):
    """
    inputs:
        `i`: array of the distances

    outputs:
        `i`: f(L)=1/(2πL²)

    Calcultes an inverse-square law distribution
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "f(L)=1/(2πL²)")

    def _fcn(self, _, inputs, outputs):
        for inp, out in zip(inputs.iter_data(), outputs.iter_data()):
            _inv_sq_law(inp, out)
        return list(outputs.iter_data())
