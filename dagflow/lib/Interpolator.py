from typing import Callable, Literal

from numba import float64, int32, njit, void
from numba.core.types import FunctionType
from numpy import exp, float_, integer, log
from numpy.typing import NDArray

from ..exception import InitializationError
from ..nodes import FunctionNode
from ..typefunctions import (
    check_has_inputs,
    check_input_dimension,
    check_input_dtype,
    check_input_shape,
    check_inputs_number,
    copy_from_input_to_output,
)


class Interpolator(FunctionNode):
    """
    inputs:
        `0` or `coarse`: array of the coarse x points
        `1` or `y`: array of the `y=f(coarse)`
        `2` or  `fine`: array of the fine x points
        `3` or `indices`: array of the indices of the coarse segments for every fine point

    outputs:
        `0` or `result`: array of the `y≈f(fine)`

    extra arguments:
        `method`: defines an interpolation method ("linear", "log"); default: `linear`
        `tolerance`: determines the accuracy with which the point will be identified
        with the segment boundary; default: `1e-10`
        `underflow`: defines the underflow strategy: `constant` or `extrapolate`;
        default: `extrapolate`
        `overflow`: defines the overflow strategy: `constant` or `extrapolate`;
        default: `extrapolate`
        `fillvalue`: defines the filling value for the `constant` strategy;
        default: `0.0`

    The node performs interpolation of the `coarse` points with `y=f(coarse)`
    to `fine` points and calculates `y≈f(fine)`.
    """

    __slots__ = ("_strategies", "_methods", "_method")

    def __init__(
        self,
        *args,
        method: Literal["linear", "log"] = "linear",
        tolerance: float = 1e-10,
        underflow: Literal["constant", "extrapolate"] = "extrapolate",
        overflow: Literal["constant", "extrapolate"] = "extrapolate",
        fillvalue: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        # TODO: implement other interpolation methods
        self._methods = {
            "linear": _linear_interpolation,
            "log": _log_interpolation,
        }
        if (mlist := self._methods.keys()) and method not in mlist:
            raise InitializationError(
                f"Argument 'method' must be in {mlist}, but given {method}!",
                node=self,
            )
        self._method = self._methods[method]
        self._strategies = {"constant": 0, "extrapolate": 1}
        self._tolerance = tolerance
        slist = self.strategies.keys()
        if underflow not in slist:
            raise InitializationError(
                f"Argument 'underflow' must be in {slist}, but given {underflow}!",
                node=self,
            )
        if overflow not in slist:
            raise InitializationError(
                f"Argument 'overflow' must be in {slist}, but given {overflow}!",
                node=self,
            )
        self._underflow = underflow
        self._overflow = overflow
        self._fillvalue = fillvalue
        self.add_input(("coarse", "y", "fine", "indices"))
        self.add_output("result")

    @property
    def methods(self) -> dict:
        return self._methods

    @property
    def strategies(self) -> dict:
        return self._strategies

    @property
    def tolerance(self) -> float:
        return self._tolerance

    @property
    def underflow(self) -> str:
        return self._underflow

    @property
    def overflow(self) -> str:
        return self._overflow

    @property
    def fillvalue(self) -> float:
        return self._fillvalue

    def _typefunc(self) -> None:
        """
        The function to determine the dtype and shape.
        Checks inputs dimension and, selects an interpolation algorithm,
        determines dtype and shape for outputs
        """
        check_inputs_number(self, 4)
        check_has_inputs(self, ("coarse", "y", "fine", "indices"))
        check_input_dimension(self, ("coarse", "y", "fine", "indices"), 1)
        check_input_dtype(self, "indices", "i")
        coarsedd = self.inputs["coarse"].dd
        check_input_shape(self, "y", coarsedd.shape)
        copy_from_input_to_output(self, "fine", 0)

    def _fcn(self, _, inputs, outputs):
        """Runs interpolation method choosen within `method` arg"""
        coarse = inputs["coarse"].data
        yc = inputs["y"].data
        fine = inputs["fine"].data
        indices = inputs["indices"].data
        out = outputs[0].data

        _interpolation(
            self._method,
            coarse,
            yc,
            fine,
            indices,
            out,
            self.tolerance,
            self.strategies[self.underflow],
            self.strategies[self.overflow],
            self.fillvalue,
        )

        if self.debug:
            return out


@njit(
    void(
        FunctionType(float64(float64, float64, float64, float64, float64)),
        float64[:],
        float64[:],
        float64[:],
        int32[:],
        float64[:],
        float64,
        int32,
        int32,
        float64,
    ),
    cache=True,
)
def _interpolation(
    method: Callable[[float, float, float, float, float], float],
    coarse: NDArray[float_],
    yc: NDArray[float_],
    fine: NDArray[float_],
    indices: NDArray[integer],
    result: NDArray[float_],
    tolerance: float,
    underflow: int,
    overflow: int,
    fillvalue: float,
) -> None:
    nseg = coarse.size - 1
    for i, j in enumerate(indices):
        if j >= nseg:  # overflow
            if overflow == 0:  # constant
                result[i] = fillvalue
            else:  # extrapolate
                result[i] = method(
                    coarse[nseg - 1],
                    coarse[nseg],
                    yc[nseg - 1],
                    yc[nseg],
                    fine[i],
                )
        elif j < 0:  # underflow
            if underflow == 0:  # constant
                result[i] = fillvalue
            else:  # extrapolate
                result[i] = method(
                    coarse[0],
                    coarse[1],
                    yc[0],
                    yc[1],
                    fine[0],
                )
        elif abs(fine[i] - coarse[j]) < tolerance:
            # get precise value from coarse
            result[i] = yc[j]
        else:
            # interpolate
            result[i] = method(
                coarse[j], coarse[j + 1], yc[j], yc[j + 1], fine[i]
            )


@njit(
    float64(
        float64,
        float64,
        float64,
        float64,
        float64,
    ),
    cache=True,
)
def _linear_interpolation(
    coarse0: float,
    coarse1: float,
    yc0: float,
    yc1: float,
    fine: float,
) -> float:
    return yc0 + (fine - coarse0) * (yc1 - yc0) / (coarse1 - coarse0)


@njit(
    float64(
        float64,
        float64,
        float64,
        float64,
        float64,
    ),
    cache=True,
)
def _log_interpolation(
    coarse0: float,
    coarse1: float,
    yc0: float,
    yc1: float,
    fine: float,
) -> float:
    return log(
        exp(yc0)
        + (fine - coarse0) * (exp(yc1) - exp(yc0)) / (coarse1 - coarse0)
    )
