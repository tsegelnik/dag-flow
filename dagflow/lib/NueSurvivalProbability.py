from numba import float64, njit, void
from numpy import empty, float_, sin, sqrt
from numpy.typing import NDArray

from ..nodes import FunctionNode
from ..typefunctions import (
    check_input_dimension,
    check_input_shape,
    copy_from_input_to_output,
)


@njit(
    void(
        float64[:],
        float64[:],
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
    L4E: NDArray[float_],
    sinSq2Theta12: float,
    sinSq2Theta13: float,
    DeltaMSq21: float,
    DeltaMSq32: float,
    alpha: float,
) -> None:
    DeltaMSq31 = DeltaMSq32 + DeltaMSq21  # |Δm²₃₁| = |Δm²₃₂| + |Δm²₂₁|
    _DeltaMSq32 = alpha * DeltaMSq31 - DeltaMSq21  # proper value of |Δm²₃₂|
    _DeltaMSq31 = alpha * DeltaMSq32 + DeltaMSq21  # proper value of |Δm²₃₁|
    _sinSqTheta12 = 0.5 * (1 - sqrt(1 - sinSq2Theta12))  # sin^2 θ_{12}
    _cosSqTheta12 = 1.0 - _sinSqTheta12  # cos^2 θ_{12}
    _cosQuTheta13 = (0.5 * (1 - sqrt(1 - sinSq2Theta13))) ** 2  # cos^4 θ_{13}

    out[:] = (
        1
        - sinSq2Theta13
        * (
            _sinSqTheta12 * sin(_DeltaMSq32 * L4E[:]) ** 2
            + _cosSqTheta12 * sin(_DeltaMSq31 * L4E[:]) ** 2
        )
        - sinSq2Theta12 * _cosQuTheta13 * sin(DeltaMSq21 * L4E[:]) ** 2
    )


class NueSurvivalProbability(FunctionNode):
    """
    inputs:
        `E`: array of the energies
        `L`: the distance
        `sinSq2Theta12`: sin²2θ₁₂
        `sinSq2Theta13`: sin²2θ₁₃
        `DeltaMSq21`: |Δm²₂₁|
        `DeltaMSq32`: |Δm²₃₂|
        `alpha`: the mass ordering constant


    outputs:
        `0` or `result`: array of probabilities

    Calcultes a survival probability for the neutrino
    """

    __slots__ = ("__buffer",)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "P(E, L, sin²2θ₁₂, sin²2θ₁₃, Δm²₂₁, Δm²₃₂, α)")
        self.add_input(
            (
                "L",
                "sinSq2Theta12",
                "sinSq2Theta13",
                "DeltaMSq21",
                "DeltaMSq32",
                "alpha",
            ),
            positional=False,
        )
        self._add_output("result")

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        check_input_dimension(
            self,
            (
                "E",
                "L",
                "sinSq2Theta12",
                "sinSq2Theta13",
                "DeltaMSq21",
                "DeltaMSq32",
                "alpha",
            ),
            1,
        )
        check_input_shape(
            self,
            (
                "L",
                "sinSq2Theta12",
                "sinSq2Theta13",
                "DeltaMSq21",
                "DeltaMSq32",
                "alpha",
            ),
            (1,),
        )
        copy_from_input_to_output(self, "E", "result")

    def _post_allocate(self):
        Edd = self.inputs["E"].dd
        self.__buffer = empty(dtype=Edd.dtype, shape=Edd.shape)

    def _fcn(self, _, inputs, outputs):
        out = outputs["result"].data
        E = inputs["E"].data
        L = inputs["L"].data[0]
        sinSq2Theta12 = inputs["sinSq2Theta12"].data[0]
        sinSq2Theta13 = inputs["sinSq2Theta13"].data[0]
        DeltaMSq21 = inputs["DeltaMSq21"].data[0]
        DeltaMSq32 = inputs["DeltaMSq32"].data[0]
        alpha = inputs["alpha"].data[0]

        self.__buffer[:] = L / 4.0 / E[:]  # common factor

        _osc_prob(
            out,
            self.__buffer,
            sinSq2Theta12,
            sinSq2Theta13,
            DeltaMSq21,
            DeltaMSq32,
            alpha,
        )
        return out
