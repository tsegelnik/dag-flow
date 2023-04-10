from numba import njit
from numpy import empty, floating, integer, issubdtype, multiply
from numpy.typing import NDArray

from ..exception import TypeFunctionError
from ..input_extra import MissingInputAddEach
from ..nodes import FunctionNode
from ..typefunctions import (
    check_has_inputs,
    check_input_dimension,
    check_input_dtype,
    check_input_shape,
)
from ..types import ShapeLike


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
    of every input within the `weight`, `ordersX` and `ordersY` (for 2 dim).

    The `dim` and `precision=dtype` of integration are chosen *automaticly*
    in the type function within the inputs.

    For 2d-integration the `ordersY` input must be connected.

    Note that the `Integrator` preallocates temporary buffer.
    For the integration algorithm the `Numba`_ package is used.

    .. _Numba: https://numba.pydata.org
    """

    __buffer: NDArray

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("missing_input_handler", MissingInputAddEach())
        super().__init__(*args, **kwargs)
        self._add_input("weights", positional=False)
        self._add_input("ordersX", positional=False)
        self._functions.update({1: self._fcn_1d, 2: self._fcn_2d})

    def _typefunc(self) -> None:
        """
        The function to determine the dtype and shape.
        Checks inputs dimension and, selects an integration algorithm,
        determines dtype and shape for outputs
        """
        check_has_inputs(self)
        check_has_inputs(self, "weights")
        input0 = self.inputs[0]
        dim = 1 if self.inputs.get("ordersY", None) is None else 2
        if (ndim := len(input0.dd.shape)) != dim:
            raise TypeFunctionError(
                f"The Integrator works only with {dim}d inputs, but the first is {ndim}d!",
                node=self,
            )
        check_input_dimension(self, (slice(None), "weights"), dim)
        check_input_shape(self, (slice(None), "weights"), input0.dd.shape)
        shape = [self.__check_orders("ordersX", input0.dd.shape[0])]
        dtype = input0.dd.dtype
        if not issubdtype(dtype, floating):
            raise TypeFunctionError(
                "The Integrator works only within `float` or `double` "
                f"precision, but given '{dtype}'!",
                node=self,
            )
        check_input_dtype(self, (slice(None), "weights"), dtype)
        if dim == 2:
            shape.append(self.__check_orders("ordersY", input0.dd.shape[1]))
        shape = tuple(shape)
        self.fcn = self._functions[dim]
        for output in self.outputs:
            output.dd.dtype = dtype
            output.dd.shape = shape

    def __check_orders(self, name: str, shape: ShapeLike):
        """
        The method checks dimension (==1) of the input `name`, type (==`integer`),
        and `sum(orders) == len(input)`
        """
        check_input_dimension(self, name, 1)
        orders = self.inputs[name]
        if not issubdtype(orders.dd.dtype, integer):
            raise TypeFunctionError(
                f"The '{name}' must be array of integers, but given '{orders.dd.dtype}'!",
                node=self,
                input=orders,
            )
        if (y := sum(orders.data)) != shape:
            raise TypeFunctionError(
                f"Orders '{name}' must be consistent with inputs len={shape}, "
                f"but given '{y}'!",
                node=self,
                input=orders,
            )
        return len(orders.dd.axes_edges) - 1

    def _post_allocate(self):
        """Allocates the `buffer` within `weights`"""
        weights = self.inputs["weights"].dd
        self.__buffer = empty(shape=weights.shape, dtype=weights.dtype)

    def _fcn_1d(self, _, inputs, outputs):
        """1d version of integration function"""
        weights = inputs["weights"].data
        ordersX = inputs["ordersX"].data
        for input, output in zip(inputs.iter_data(), outputs.iter_data()):
            multiply(input, weights, out=self.__buffer)
            _integrate1d(output, self.__buffer, ordersX)
        if self.debug:
            return list(outputs.iter_data())

    def _fcn_2d(self, _, inputs, outputs):
        """2d version of integration function"""
        weights = inputs["weights"].data # (n, m)
        ordersX = inputs["ordersX"].data # (n, )
        ordersY = inputs["ordersY"].data # (m, )
        for input, output in zip(inputs.iter_data(), outputs.iter_data()):
            multiply(input, weights, out=self.__buffer)
            _integrate2d(output, self.__buffer, ordersX, ordersY)
        if self.debug:
            return list(outputs.iter_data())
