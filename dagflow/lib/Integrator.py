from numba import njit
from numpy import empty, floating, integer, multiply
from numpy.typing import NDArray

from ..exception import TypeFunctionError
from ..input_extra import MissingInputAddEach
from ..nodes import FunctionNode
from ..typefunctions import (
    check_has_inputs,
    check_input_dimension,
    check_input_dtype,
    check_input_edges_dim,
    check_input_edges_equivalence,
    check_input_shape,
    check_input_subtype,
    check_output_subtype,
)
from ..types import ShapeLike


@njit(cache=True)
def _integrate1d(result: NDArray, data: NDArray, ordersX: NDArray):
    """
    Summing up `data` within `ordersX` and puts the result into `result`.
    The 1-dimensional version of integration.
    """
    iprev = 0
    for i, order in enumerate(ordersX):
        inext = iprev + order
        result[i] = data[iprev:inext].sum()
        iprev = inext


@njit(cache=True)
def _integrate2d(
    result: NDArray, data: NDArray, ordersX: NDArray, ordersY: NDArray
):
    """
    Summing up `data` within `ordersX` and `ordersY` and then
    puts the result into `result`. The 2-dimensional version of integration.
    """
    iprev = 0
    for i, orderx in enumerate(ordersX):
        inext = iprev + orderx
        jprev = 0
        for j, ordery in enumerate(ordersY):
            jnext = jprev + ordery
            result[i, j] = data[iprev:inext, jprev:jnext].sum()
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

    __slots__ = ("__buffer",)

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
        check_input_subtype(self, input0, floating)
        dtype = input0.dd.dtype
        check_input_dtype(self, (slice(None), "weights"), dtype)

        edgeslenX, edgesX = self.__check_orders("ordersX", input0.dd.shape[0])
        shape = [edgeslenX]
        edges = [edgesX]
        if dim == 2:
            edgeslenY, edgesY = self.__check_orders(
                "ordersY", input0.dd.shape[1]
            )
            shape.append(edgeslenY)
            edges.append(edgesY)
        check_input_edges_equivalence(self, slice(None), edges)

        shape = tuple(shape)
        self.fcn = self._functions[dim]
        for output in self.outputs:
            output.dd.dtype = dtype
            output.dd.shape = shape
            output.dd.axes_edges = edges
            # TODO: copy axes_nodes?

    def __check_orders(self, name: str, shape: ShapeLike) -> tuple:
        """
        The method checks dimension (==1) of the input `name`, type (==`integer`),
        and `sum(orders) == len(input)`
        """
        check_input_dimension(self, name, 1)
        orders = self.inputs[name]
        check_output_subtype(self, orders, integer)
        if (y := sum(orders.data)) != shape:
            raise TypeFunctionError(
                f"Orders '{name}' must be consistent with inputs len={shape}, "
                f"but given '{y}'!",
                node=self,
                input=orders,
            )
        check_input_edges_dim(self, name, 1)
        edges = orders.dd.axes_edges[0]
        return edges.dd.shape[0] - 1, edges

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
        weights = inputs["weights"].data  # (n, m)
        ordersX = inputs["ordersX"].data  # (n, )
        ordersY = inputs["ordersY"].data  # (m, )
        for input, output in zip(inputs.iter_data(), outputs.iter_data()):
            multiply(input, weights, out=self.__buffer)
            _integrate2d(output, self.__buffer, ordersX, ordersY)
        if self.debug:
            return list(outputs.iter_data())
