from numba import float64, njit, void
from numpy import float_, sin, sqrt
from numpy.typing import NDArray

from ..nodes import FunctionNode
from ..typefunctions import copy_from_input_to_output


@njit(
    void(
        float64[:],
        float64[:],
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
    ),
    cache=True,
)
def _osc_prob(
    out: NDArray[float_],
    E: NDArray[float_],
    L: float,
    sinSq2Theta13: float,
    DeltaMSq32: float,
    sinSq2Theta12: float,
    DeltaMSq21: float,
    DeltaMSq31: float,
    alpha: float,
) -> None:
    _DeltaMSq32 = alpha * DeltaMSq31 - DeltaMSq21  # proper value of |Δm²₃₂|
    _DeltaMSq31 = alpha * DeltaMSq32 + DeltaMSq21  # proper value of |Δm²₃₁|
    _sinSqTheta12 = 0.5 * (1 - sqrt(1 - sinSq2Theta12))  # sin^2 θ_{12}
    _cosSqTheta12 = 1.0 - _sinSqTheta12  # cos^2 θ_{12}
    _cosQuTheta13 = (0.5 * (1 - sqrt(1 - sinSq2Theta13))) ** 2  # cos^4 θ_{13}

    L4E = L / 4.0 / E  # common multiplier
    out[:] = (
        1
        - sinSq2Theta13
        * (
            _sinSqTheta12 * sin(_DeltaMSq32 * L4E) ** 2
            + _cosSqTheta12 * sin(_DeltaMSq31 * L4E) ** 2
        )
        - sinSq2Theta12 * _cosQuTheta13 * sin(DeltaMSq21 * L4E) ** 2
    )


class OscProb(FunctionNode):
    """
    inputs:
        `E`: array of the energies

    outputs:
        `0` or `result`: array of probabilities

    extra arguments:
        `L`: the distance
        `sinSq2Theta13`: sin²2θ₁₃
        `DeltaMSq32`: |Δm²₃₂|
        `sinSq2Theta12`: sin²2θ₁₂
        `DeltaMSq21`: |Δm²₂₁|
        `DeltaMSq31`: |Δm²₃₁|
        `alpha`: the mass ordering constant

    Calcultes a probability of the neutrino oscillations
    """

    def __init__(
        self,
        *args,
        L: float,
        sinSq2Theta13: float,
        DeltaMSq32: float,
        sinSq2Theta12: float,
        DeltaMSq21: float,
        DeltaMSq31: float,
        alpha: float,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._labels.setdefault(
            "mark", "P(E, L, sin²2θ₁₃, Δm²₃₂, sin²2θ₁₂, Δm²₂₁, Δm²₃₁, α)"
        )
        self._L = L
        self._sinSqTheta12 = sinSq2Theta12
        self._sinSqTheta13 = sinSq2Theta13
        self._DeltaMSq21 = DeltaMSq21
        self._DeltaMSq32 = DeltaMSq32
        self._DeltaMSq31 = DeltaMSq31
        self._alpha = alpha

    @property
    def L(self) -> float:
        return self._L

    @property
    def sinSq2Theta12(self) -> float:
        return self._sinSqTheta12

    @property
    def sinSq2Theta13(self) -> float:
        return self._sinSqTheta13

    @property
    def DeltaMSq21(self) -> float:
        return self._DeltaMSq21

    @property
    def DeltaMSq32(self) -> float:
        return self._DeltaMSq32

    @property
    def DeltaMSq31(self) -> float:
        return self._DeltaMSq31

    @property
    def alpha(self) -> float:
        return self._alpha

    def _fcn(self, _, inputs, outputs):
        out = outputs["result"].data
        E = inputs["E"].data
        _osc_prob(
            out,
            E,
            self.L,
            self.sinSq2Theta13,
            self.DeltaMSq32,
            self.sinSq2Theta12,
            self.DeltaMSq21,
            self.DeltaMSq31,
            self.alpha,
        )
        return out

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        copy_from_input_to_output(self, "E", "result")
