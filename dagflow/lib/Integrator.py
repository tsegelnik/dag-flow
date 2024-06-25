from __future__ import annotations

from typing import TYPE_CHECKING

from numba import njit
from numpy import empty, floating, integer, multiply

from ..exception import TypeFunctionError
from ..inputhandler import MissingInputAddPair
from ..nodes import FunctionNode
from ..typefunctions import (
    check_has_inputs,
    check_input_dimension,
    check_input_dtype,
    check_input_edges_dim,
    check_input_shape,
    check_input_subtype,
)
from ..types import ShapeLike

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..input import Input


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
def _integrate2d(result: NDArray, data: NDArray, ordersX: NDArray, ordersY: NDArray):
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


@njit(cache=True)
def _integrate2to1d(result: NDArray, data: NDArray, orders: NDArray):
    """
    Summing up `data` within `orders` and then puts the result into `result`.
    The 21-dimensional version of integration, where y dimension is dropped.

    .. note:: Note that the x dimension drop uses a matrix transpose before
    """
    iprev = 0
    for i, orderx in enumerate(orders):
        inext = iprev + orderx
        result[i] = data[iprev:inext, :].sum()
        iprev = inext


class Integrator(FunctionNode):
    """
    self.inputs:
        `i`: points to integrate
        `ordersX`: array with orders to integrate by x-axis (1d array)
        `weights`: array with weights (1d or 2d array)
        `ordersY` (optional): array with orders to integrate by y-axis (1d array)

    self.outputs:
        `i`: result of integration

    extra arguments:
        `dropdim`: If `True` drops dimension in a 2d integration by axis with
        only one bin; default: `True`

    The `Integrator` node performs integration (summation)
    of every input within the `weight`, `ordersX` and `ordersY` (for 2 dim).

    The `dim` and `precision=dtype` of integration are chosen *automaticly*
    in the type function within the self.inputs.

    For 2d-integration the `ordersY` input must be connected.
    If any dimension has only one bin, the integrator may drop this dimension and
    return 1d array.

    Note that the `Integrator` preallocates temporary buffer.
    For the integration algorithm the `Numba`_ package is used.

    .. _Numba: https://numba.pydata.org
    """

    __slots__ = ("__buffer", "_dropdim", "_ordersX", "_ordersY", "_weights")

    __buffer: NDArray
    _dropdim: bool
    _ordersX: Input
    _ordersY: Input
    _weights: Input

    def __init__(self, *args, dropdim: bool = True, **kwargs):
        kwargs.setdefault("missing_input_handler", MissingInputAddPair())
        super().__init__(*args, **kwargs, allowed_kw_inputs=("ordersX", "ordersY", "weights"))
        self._dropdim = dropdim
        self._weights = self._add_input("weights", positional=False)
        self._ordersX = self._add_input("ordersX", positional=False)
        self._functions.update(
            {
                1: self._fcn_1d,
                2: self._fcn_2d,
                210: self._fcn_21d_x,
                211: self._fcn_21d_y,
            }
        )
        self.labels.setdefault("mark", "∫")

    @property
    def dropdim(self) -> bool:
        return self._dropdim

    def _typefunc(self) -> None:
        """
        The function to determine the dtype and shape.
        Checks self.inputs dimension and, selects an integration algorithm,
        determines dtype and shape for self.outputs
        """
        if len(self.inputs) == 0:
            return
        check_has_inputs(self, "weights")

        input0 = self.inputs[0]
        self._ordersY = self.inputs.get("ordersY", None)
        dim = 1 if self._ordersY is None else 2
        if (ndim := len(input0.dd.shape)) != dim:
            raise TypeFunctionError(
                f"The Integrator works only with {dim}d self.inputs, but the first is {ndim}d!",
                node=self,
            )
        check_input_dimension(self, (slice(None), "weights"), dim)
        check_input_shape(self, (slice(None), "weights"), input0.dd.shape)
        check_input_subtype(self, 0, floating)
        dtype = input0.dd.dtype
        check_input_dtype(self, (slice(None), "weights"), dtype)

        edgeslenX, edgesX = self.__check_orders("ordersX", input0.dd.shape[0])
        if dim == 2:
            edgeslenY, edgesY = self.__check_orders("ordersY", input0.dd.shape[1])
            if self.dropdim and edgeslenY == 2:  # drop Y dimension
                shape = (edgeslenX - 1,)
                edges = [edgesX]
                self.fcn = self._functions[211]
            elif self.dropdim and edgeslenX == 2:  # drop X dimension
                shape = (edgeslenY - 1,)
                edges = [edgesY]
                self.fcn = self._functions[210]
            else:
                shape = (edgeslenX - 1, edgeslenY - 1)
                edges = [edgesX, edgesY]
                self.fcn = self._functions[2]
        else:
            shape = (edgeslenX - 1,)
            edges = [edgesX]
            self.fcn = self._functions[1]

        for output in self.outputs:
            output.dd.dtype = dtype
            output.dd.shape = shape
            output.dd.axes_edges = edges
            # TODO: copy axes_meshes?

    def __check_orders(self, name: str, shape: ShapeLike) -> tuple:
        """
        The method checks dimension (==1) of the input `name`, type (==`integer`),
        and `sum(orders) == len(input)`
        """
        check_input_dimension(self, name, 1)
        check_input_subtype(self, name, integer)
        orders = self.inputs[name]
        if (y := sum(orders.data)) != shape:
            raise TypeFunctionError(
                (
                    f"Orders '{name}' must be consistent with self.inputs len={shape}, "
                    f"but given '{y}'!"
                ),
                node=self,
                input=orders,
            )
        check_input_edges_dim(self, name, 1)
        edges = orders.dd.axes_edges[0]
        return edges.dd.shape[0], edges

    def _post_allocate(self):
        """Allocates the `buffer` within `weights`"""
        weights = self._weights.dd
        self.__buffer = empty(shape=weights.shape, dtype=weights.dtype)

    def _fcn_1d(self):
        """1d version of integration function"""
        weights = self._weights.data
        ordersX = self._ordersX.data
        for _input, output in zip(self.inputs.iter_data(), self.outputs.iter_data()):
            multiply(_input, weights, out=self.__buffer)
            _integrate1d(output, self.__buffer, ordersX)

    def _fcn_2d(self):
        """2d version of integration function"""
        weights = self._weights.data  # (n, m)
        ordersX = self._ordersX.data  # (n, )
        ordersY = self._ordersY.data  # (m, )
        for _input, output in zip(self.inputs.iter_data(), self.outputs.iter_data()):
            multiply(_input, weights, out=self.__buffer)
            _integrate2d(output, self.__buffer, ordersX, ordersY)

    def _fcn_21d_x(self):
        """21d version of integration function where x-axis is dropped"""
        weights = self._weights.data  # (1, m)
        ordersY = self._ordersY.data  # (m, )
        for _input, output in zip(self.inputs.iter_data(), self.outputs.iter_data()):
            multiply(_input, weights, out=self.__buffer)
            _integrate2to1d(output, self.__buffer.T, ordersY)

    def _fcn_21d_y(self):
        """21d version of integration function where y-axis is dropped"""
        weights = self._weights.data  # (m, 1)
        ordersX = self._ordersX.data  # (m, )
        for _input, output in zip(self.inputs.iter_data(), self.outputs.iter_data()):
            multiply(_input, weights, out=self.__buffer)
            _integrate2to1d(output, self.__buffer, ordersX)
