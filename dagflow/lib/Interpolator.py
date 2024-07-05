from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

from numba import float64, int32, njit, void
from numba.core.types import FunctionType
from numpy import double, exp, integer, log

from ..exception import InitializationError
from ..node import Node
from ..typefunctions import (
    assign_output_axes_from_inputs,
    check_has_inputs,
    check_input_dtype,
    check_input_shape,
    check_inputs_number,
    copy_from_input_to_output,
)

if TYPE_CHECKING:
    from typing import Callable, Literal

    from numpy.typing import NDArray

    from ..input import Input
    from ..output import Output


class ExtrapolationStrategy(IntEnum):
    constant = 0
    nearestedge = 1
    extrapolate = 2


class Interpolator(Node):
    """
    self.inputs:
        `0` or `y`: array of the `y=f(coarse)`
        `coarse`: array of the coarse x points
        `fine`: array of the fine x points
        `indices`: array of the indices of the coarse segments for every fine point

    self.outputs:
        `0` or `result`: array of the `y≈f(fine)`

    extra arguments:
        `method`: defines an interpolation method ("linear", "log", "logx", "exp");
        default: `linear`
        `tolerance`: determines the accuracy with which the point will be identified
        with the segment boundary; default: `1e-10`
        `underflow`: defines the underflow strategy: `constant`, `nearestedge`, or `extrapolate`;
        default: `extrapolate`
        `overflow`: defines the overflow strategy: `constant`, `nearestedge`, or `extrapolate`;
        default: `extrapolate`
        `fillvalue`: defines the filling value for the `constant` strategy;
        default: `0.0`

    The node performs interpolation of the `coarse` points with `y=f(coarse)`
    to `fine` points and calculates `y≈f(fine)`.
    """

    __slots__ = (
        "_strategies",
        "_methods",
        "_method",
        "_tolerance",
        "_underflow",
        "_overflow",
        "_fillvalue",
        "_y",
        "_coarse",
        "_fine",
        "_indices",
        "_result",
    )

    _y: Input
    _coarse: Input
    _fine: Input
    _indices: Input
    _result: Output

    def __init__(
        self,
        *args,
        method: Literal["linear", "log", "logx", "exp"] = "linear",
        tolerance: float = 1e-10,
        underflow: Literal["constant", "nearestedge", "extrapolate"] = "extrapolate",
        overflow: Literal["constant", "nearestedge", "extrapolate"] = "extrapolate",
        fillvalue: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs, allowed_kw_inputs=("y", "coarse", "fine", "indices"))
        self._labels.setdefault("mark", "~")
        self._methods = {
            "linear": _linear_interpolation,
            "log": _log_interpolation,
            "logx": _logx_interpolation,
            "exp": _exp_interpolation,
        }
        if (mlist := self._methods.keys()) and method not in mlist:
            raise InitializationError(
                f"Argument 'method' must be in {mlist}, but given {method}!",
                node=self,
            )
        self._method = self._methods[method]
        self._strategies = {"constant": 0, "nearestedge": 1, "extrapolate": 2}
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
        # inputs/outputs
        self._y = self._add_input("y")
        self._coarse = self._add_input("coarse", positional=False)
        self._fine = self._add_input("fine", positional=False)
        self._indices = self._add_input("indices", positional=False)
        self._result = self._add_output("result")

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
        Checks self.inputs dimension and, selects an interpolation algorithm,
        determines dtype and shape for self.outputs
        """
        check_inputs_number(self, 1)
        check_has_inputs(self, ("coarse", "y", "fine", "indices"))
        check_input_dtype(self, "indices", "i")
        check_input_shape(self, "y", self._coarse.dd.shape)
        check_input_shape(self, "fine", self._indices.dd.shape)
        copy_from_input_to_output(self, "fine", "result")
        if self._fine.dd.dim == 1:
            assign_output_axes_from_inputs(
                self, "fine", "result", assign_meshes=True, ignore_assigned=False
            )
        else:
            # TODO: add a choice of what axis to overwrite
            assign_output_axes_from_inputs(
                self,
                "fine",
                "result",
                assign_meshes=True,
                ignore_assigned=False,
                overwrite_assigned=True,
                ignore_inconsistent_number_of_meshes=True,
            )

    def _fcn(self):
        """Runs interpolation method chosen within `method` arg"""
        coarse = self._coarse.data.ravel()
        yc = self._y.data.ravel()
        fine = self._fine.data.ravel()
        indices = self._indices.data.ravel()
        out = self._result.data.ravel()

        sortedindices = coarse.argsort()  # indices to sort the arrays
        _interpolation(
            self._method,
            coarse[sortedindices],
            yc[sortedindices],
            fine,
            indices,
            out,
            self.tolerance,
            self.strategies[self.underflow],
            self.strategies[self.overflow],
            self.fillvalue,
        )


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
    coarse: NDArray[double],
    yc: NDArray[double],
    fine: NDArray[double],
    indices: NDArray[integer],
    result: NDArray[double],
    tolerance: float,
    underflow: int,
    overflow: int,
    fillvalue: float,
) -> None:
    nseg = coarse.size - 1
    for i, j in enumerate(indices):
        if abs(fine[i] - coarse[j]) < tolerance:
            # get precise value from coarse
            result[i] = yc[j]
        elif j > nseg:  # overflow
            if overflow == ExtrapolationStrategy.constant:  # constant
                result[i] = fillvalue
            elif overflow == ExtrapolationStrategy.nearestedge:  # nearestedge
                result[i] = yc[nseg]
            else:  # extrapolate
                result[i] = method(
                    coarse[nseg - 1],
                    coarse[nseg],
                    yc[nseg - 1],
                    yc[nseg],
                    fine[i],
                )
        elif j <= 0:  # underflow
            if underflow == ExtrapolationStrategy.constant:  # constant
                result[i] = fillvalue
            elif underflow == ExtrapolationStrategy.nearestedge:  # nearestedge
                result[i] = yc[0]
            else:  # extrapolate
                result[i] = method(
                    coarse[0],
                    coarse[1],
                    yc[0],
                    yc[1],
                    fine[i],
                )
        else:
            # interpolate
            result[i] = method(coarse[j - 1], coarse[j], yc[j - 1], yc[j], fine[i])


@njit(cache=True)
def _linear_interpolation(
    coarse0: float,
    coarse1: float,
    yc0: float,
    yc1: float,
    fine: float,
) -> float:
    return yc0 + (fine - coarse0) * (yc1 - yc0) / (coarse1 - coarse0)


@njit(cache=True)
def _log_interpolation(
    coarse0: float,
    coarse1: float,
    yc0: float,
    yc1: float,
    fine: float,
) -> float:
    return log(exp(yc0) + (fine - coarse0) * (exp(yc1) - exp(yc0)) / (coarse1 - coarse0))


@njit(cache=True)
def _logx_interpolation(
    coarse0: float,
    coarse1: float,
    yc0: float,
    yc1: float,
    fine: float,
) -> float:
    return yc0 + log(fine / coarse0) * (yc1 - yc0) / log(coarse1 / coarse0)


@njit(cache=True)
def _exp_interpolation(
    coarse0: float,
    coarse1: float,
    yc0: float,
    yc1: float,
    fine: float,
) -> float:
    return yc0 * exp((coarse0 - fine) * log(yc0 / yc1) / (coarse1 - coarse0))
