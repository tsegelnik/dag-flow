from typing import Literal

from numba import njit
from numpy import floating, integer, issubdtype, multiply, zeros
from numpy.typing import NDArray

from ..exception import InitializationError, TypeFunctionError
from ..input_extra import MissingInputAddEach
from ..nodes import FunctionNode
from ..typefunctions import (
    check_has_inputs,
    check_input_dimension,
    check_input_dtype,
    check_input_shape,
)


@njit(cache=True)
def _integrate1d(data: NDArray, weighted: NDArray, ordersX: NDArray):
    """
    Summing up `weighted` within `ordersX` and puts the result into `data`.
    The 1-dimensional version of integration.
    """
    iprev = 0
    for i, order in enumerate(ordersX):
        inext = iprev + order
        data[i] = weighted[iprev:inext].sum()
        iprev = inext


@njit(cache=True)
def _integrate2d(
    data: NDArray, weighted: NDArray, ordersX: NDArray, ordersY: NDArray
):
    """
    Summing up `weighted` within `ordersX` and `ordersY` and then
    puts the result into `data`. The 2-dimensional version of integration.
    """
    iprev = 0
    for i, orderx in enumerate(ordersX):
        inext = iprev + orderx
        jprev = 0
        for j, ordery in enumerate(ordersY):
            jnext = jprev + ordery
            data[i, j] = weighted[iprev:inext, jprev:jnext].sum()
            jprev = jnext
        iprev = inext


class Integrator(FunctionNode):
    """
    The `Integrator` node performs integration (summation)
    of every input within the `weight`, `ordersX` and `ordersY` (for `2d` mode).

    The `Integrator` has two modes: `1d` and `2d`.
    The `mode` must be set in the constructor, while `precision=dtype`
    of integration is chosen *automaticly* in the type function.

    For `2d` integration the `ordersY` input must be connected.

    Note that the `Integrator` preallocates temporary buffer.
    For the integration algorithm the `Numba`_ package is used.

    .. _Numba: https://numba.pydata.org
    """

    def __init__(self, *args, mode: Literal["1d", "2d"], **kwargs):
        kwargs.setdefault("missing_input_handler", MissingInputAddEach())
        super().__init__(*args, **kwargs)
        if mode not in {"1d", "2d"}:
            raise InitializationError(
                f"Argument `mode` must be '1d' or '2d', but given '{mode}'!",
                node=self,
            )
        self._mode = mode
        if self._mode == "2d":
            self._add_input("ordersY", positional=False)
        self._add_input("weights", positional=False)
        self._add_input("ordersX", positional=False)
        self._functions.update({"1d": self._fcn_1d, "2d": self._fcn_2d})

    @property
    def mode(self) -> str:
        return self._mode

    def _typefunc(self) -> None:
        """
        The function to determine the dtype and shape.
        Checks inputs dimension and, selects an integration algorithm,
        determines dtype and shape for outputs
        """
        check_has_inputs(self)
        check_has_inputs(self, ("ordersX", "weights"))
        input0 = self.inputs[0]
        ndim = len(input0.dd.shape)
        if ndim != int(self.mode[:1]):
            raise TypeFunctionError(
                f"The Integrator works only with {self.mode} inputs, but one has ndim={ndim}!",
                node=self,
            )
        check_input_dimension(self, (slice(None), "weights"), ndim)
        check_input_dimension(self, "ordersX", 1)
        check_input_shape(self, (slice(None), "weights"), input0.dd.shape)
        ordersX = self.inputs["ordersX"]
        if not issubdtype(ordersX.dd.dtype, integer):
            raise TypeFunctionError(
                "The `ordersX` must be array of integers, but given '{ordersX.dd.dtype}'!",
                node=self,
                input=ordersX,
            )
        dtype = input0.dd.dtype
        if not issubdtype(dtype, floating):
            raise TypeFunctionError(
                "The Integrator works only within `float` or `double` "
                f"precision, but given '{dtype}'!",
                node=self,
            )
        check_input_dtype(self, (slice(None), "weights"), dtype)
        if sum(ordersX.data) != input0.dd.shape[0]:
            raise TypeFunctionError(
                "ordersX must be consistent with inputs shape, "
                f"but given {ordersX.data=} and {input0.dd.shape=}!",
                node=self,
                input=ordersX,
            )
        if self.mode == "2d":
            check_has_inputs(self, "ordersY")
            check_input_dimension(self, "ordersY", 1)
            ordersY = self.inputs["ordersY"]
            if not issubdtype(ordersY.dd.dtype, integer):
                raise TypeFunctionError(
                    "The `ordersY` must be array of integers, but given '{ordersY.dd.dtype}'!",
                    node=self,
                    input=ordersY,
                )
            if sum(ordersY.data) != input0.dd.shape[1]:
                raise TypeFunctionError(
                    "ordersY must be consistent with inputs shape, "
                    f"but given {ordersY.data=} and {input0.dd.shape=}!",
                    node=self,
                    input=ordersX,
                )
        self.fcn = self._functions[self.mode]
        for output in self.outputs:
            output.dd.dtype = dtype
            output.dd.shape = input0.dd.shape

    def _post_allocate(self):
        """Allocates the `buffer` within `weights`"""
        weights = self.inputs["weights"]
        self.__buffer = zeros(shape=weights.dd.shape, dtype=weights.dd.dtype)

    def _fcn_1d(self, _, inputs, outputs):
        """1d version of integration function"""
        weights = inputs["weights"].data
        ordersX = inputs["ordersX"].data
        for input, output in zip(inputs.iter_data(), outputs.iter_data()):
            multiply(input, weights, out=self.__buffer)
            _integrate1d(output, self.__buffer, ordersX)
        if self.debug:
            return [outputs.iter_data()]

    def _fcn_2d(self, _, inputs, outputs):
        """2d version of integration function"""
        weights = inputs["weights"].data
        ordersX = inputs["ordersX"].data
        ordersY = inputs["ordersY"].data
        for input, output in zip(inputs.iter_data(), outputs.iter_data()):
            multiply(input, weights, out=self.__buffer)
            _integrate2d(output, self.__buffer, ordersX, ordersY)
        if self.debug:
            return [outputs.iter_data()]
