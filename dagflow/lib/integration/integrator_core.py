from __future__ import annotations

from typing import TYPE_CHECKING

from numba import njit
from numpy import empty, floating, integer, multiply

from ...core.exception import TypeFunctionError
from ...core.input_handler import MissingInputAddPair
from ...core.type_functions import (
    check_has_inputs,
    check_input_dimension,
    check_input_dtype,
    check_input_edges_dim,
    check_input_shape,
    check_input_subtype,
)
from ..abstract import OneToOneNode

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ...core.input import Input
    from ...types import ShapeLike


@njit(cache=True)
def _integrate1d(result: NDArray, data: NDArray, orders_x: NDArray):
    """Summing up `data` within `orders_x` and puts the result into `result`.

    The 1-dimensional version of integration.
    """
    iprev = 0
    for i, order in enumerate(orders_x):
        inext = iprev + order
        result[i] = data[iprev:inext].sum()
        iprev = inext


@njit(cache=True)
def _integrate2d(result: NDArray, data: NDArray, orders_x: NDArray, orders_y: NDArray):
    """Summing up `data` within `orders_x` and `orders_y` and then puts the
    result into `result`.

    The 2-dimensional version of integration.
    """
    iprev = 0
    for i, orderx in enumerate(orders_x):
        inext = iprev + orderx
        jprev = 0
        for j, ordery in enumerate(orders_y):
            jnext = jprev + ordery
            result[i, j] = data[iprev:inext, jprev:jnext].sum()
            jprev = jnext
        iprev = inext


@njit(cache=True)
def _integrate2to1d(result: NDArray, data: NDArray, orders: NDArray):
    """Summing up `data` within `orders` and then puts the result into
    `result`. The 21-dimensional version of integration, where y dimension is
    dropped.

    .. note:: Note that the x dimension drop uses a matrix transpose before
    """
    iprev = 0
    for i, orderx in enumerate(orders):
        inext = iprev + orderx
        result[i] = data[iprev:inext, :].sum()
        iprev = inext


class IntegratorCore(OneToOneNode):
    """
    self.inputs:
        `i`: function (computed at integration nodes) to integrate
        `orders_x`: array with orders to integrate by x-axis (1d array) 
        `weights`: array with weights (1d or 2d array)
        `orders_y` (optional): array with orders to integrate by y-axis (1d array)

    self.outputs:
        `i`: result of integration

    extra arguments:
        `dropdim`: If `True` drops dimension in a 2d integration by axis with
        only one bin; default: `True`

    The `IntegratorCore` node performs integration (summation)
    of every input within the `weight`, `orders_x` and `orders_y` (for 2 dim).

    The `dim` and `precision=dtype` of integration are chosen *automaticly*
    in the type function within the self.inputs.

    For 2d-integration the `orders_y` input must be connected.
    If any dimension has only one bin, the integrator may drop this dimension and
    return 1d array.

    Note that the `IntegratorCore` preallocates temporary buffer.
    For the integration algorithm the `Numba`_ package is used.

    .. _Numba: https://numba.pydata.org
    """

    __slots__ = (
        "__buffer",
        "_dropdim",
        "_orders_x_input",
        "_orders_y_input",
        "_weights_input",
        "_orders_x",
        "_orders_y",
        "_weights",
    )

    __buffer: NDArray
    _dropdim: bool
    _orders_x_input: Input
    _orders_y_input: Input | None
    _weights_input: Input
    _orders_x: NDArray
    _orders_y: NDArray | None
    _weights: NDArray

    def __init__(self, *args, dropdim: bool = True, ndim: Literal[1, 2] | None = None, **kwargs):
        kwargs.setdefault("missing_input_handler", MissingInputAddPair())
        super().__init__(*args, **kwargs, allowed_kw_inputs=("orders_x", "orders_y", "weights"))
        self._dropdim = dropdim
        self._weights_input = self._add_input("weights", positional=False)
        self._orders_x_input = self._add_input("orders_x", positional=False)
        if ndim == 2:
            self._orders_y_input = self._add_input("orders_y", positional=False)
        self._functions_dict.update(
            {
                1: self._fcn_1d,
                2: self._fcn_2d,
                210: self._fcn_21d_x,
                211: self._fcn_21d_y,
            }
        )
        self.labels.setdefault("mark", "∫")

        self._orders_y_input = None
        self._orders_y = None

    @property
    def dropdim(self) -> bool:
        return self._dropdim

    def _typefunc(self) -> None:
        """The function to determine the dtype and shape.

        Checks self.inputs dimension and, selects an integration
        algorithm, determines dtype and shape for self.outputs
        """
        if len(self.inputs) == 0:
            return
        check_has_inputs(self, "weights")

        input0 = self.inputs[0]
        self._orders_y_input = self.inputs.get("orders_y", None)
        dim = 1 if self._orders_y_input is None else 2
        if (ndim := len(input0.dd.shape)) != dim:
            raise TypeFunctionError(
                f"The IntegratorCore works only with {dim}d self.inputs, but the first is {ndim}d!",
                node=self,
            )
        check_input_dimension(self, (slice(None), "weights"), dim)
        check_input_shape(self, (slice(None), "weights"), input0.dd.shape)
        check_input_subtype(self, 0, dtype=floating)
        dtype = input0.dd.dtype
        check_input_dtype(self, (slice(None), "weights"), dtype=dtype)

        edgeslenX, edgesX = self.__check_orders_input("orders_x", input0.dd.shape[0])
        if dim == 2:
            edgeslenY, edgesY = self.__check_orders_input("orders_y", input0.dd.shape[1])
            if self.dropdim and edgeslenY == 2:  # drop Y dimension
                shape = (edgeslenX - 1,)
                edges = [edgesX]
                self.function = self._functions_dict[211]
            elif self.dropdim and edgeslenX == 2:  # drop X dimension
                shape = (edgeslenY - 1,)
                edges = [edgesY]
                self.function = self._functions_dict[210]
            else:
                shape = (edgeslenX - 1, edgeslenY - 1)
                edges = [edgesX, edgesY]
                self.function = self._functions_dict[2]
        else:
            shape = (edgeslenX - 1,)
            edges = [edgesX]
            self.function = self._functions_dict[1]

        for output in self.outputs:
            output.dd.dtype = dtype
            output.dd.shape = shape
            output.dd.axes_edges = edges
            # TODO: copy axes_meshes?

    def __check_orders_input(self, name: str, shape: ShapeLike) -> tuple:
        """The method checks dimension (==1) of the input `name`, type
        (==`integer`), and `sum(orders) == len(input)`"""
        check_input_dimension(self, name, 1)
        check_input_subtype(self, name, dtype=integer)
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
        super()._post_allocate()
        weights = self._weights_input.dd
        self.__buffer = empty(shape=weights.shape, dtype=weights.dtype)

        self._weights = self._weights_input.data_unsafe
        self._orders_x = self._orders_x_input.data_unsafe
        self._orders_y = self._orders_y_input.data_unsafe if self._orders_y_input else None

    def _fcn_1d(self):
        """1d version of integration function."""
        for callback in self._input_nodes_callbacks:
            callback()

        for input, output in self._input_output_data:
            multiply(input, self._weights, out=self.__buffer)
            _integrate1d(output, self.__buffer, self._orders_x)

    def _fcn_2d(self):
        """2d version of integration function."""
        for callback in self._input_nodes_callbacks:
            callback()

        # weights - (n, m)
        # orders_x - (n, )
        # orders_y - (m, )
        for input, output in self._input_output_data:
            multiply(input, self._weights, out=self.__buffer)
            _integrate2d(output, self.__buffer, self._orders_x, self._orders_y)

    def _fcn_21d_x(self):
        """21d version of integration function where x-axis is dropped."""
        for callback in self._input_nodes_callbacks:
            callback()

        # weights - (1, m)
        # orders_y - (m, )
        for input, output in self._input_output_data:
            multiply(input, self._weights, out=self.__buffer)
            _integrate2to1d(output, self.__buffer.T, self._orders_y)

    def _fcn_21d_y(self):
        """21d version of integration function where y-axis is dropped."""
        for callback in self._input_nodes_callbacks:
            callback()

        # weights - (m, 1)
        # orders_x - (m, )
        for input, output in self._input_output_data:
            multiply(input, self._weights, out=self.__buffer)
            _integrate2to1d(output, self.__buffer, self._orders_x)
