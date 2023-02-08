from numba import jit
from numpy import multiply, zeros, floating, issubdtype, integer
from numpy.typing import NDArray

from ..exception import TypeFunctionError
from ..input_extra import MissingInputAddEach
from ..nodes import FunctionNode
from ..typefunctions import (
    check_has_input,
    check_has_inputs,
    check_input_dimension,
    check_input_dtype,
)


@jit(nopython=True)
def _integrate1d(data: NDArray, weighted: NDArray, orders: NDArray):
    """
    Summing up `weighted` within `orders` and puts the result into `data`.
    The 1-dimensional version.
    """
    iprev = 0
    i = 0
    for order in orders:
        if order == 0:
            continue
        inext = iprev + order
        data[i] = weighted[iprev:inext].sum()
        iprev = inext
        i += 1


@jit(nopython=True)
def _integrate2d(data: NDArray, weighted: NDArray, orders: NDArray):
    """
    Summing up `weighted` within `orders` and puts the result into `data`
    The 2-dimensional version, so all the arrays must be 2d.

    .. note:: `Numba`_ doesn't like arrays of elements with different size,
        so arguments must have the elements with the same size.
        This happens due to typing: arrays with elements of different sizes
        have `dtype=object`, but `Numba`_ doesn't like the `object` type
        and works only with numeric and sequence types (including `Numpy`_ types)

    .. _Numba: https://numba.pydata.org
    .. _Numpy: https://numpy.org
    """
    iprev = 0
    i = 0
    for orderx in orders[0]:
        if orderx == 0:
            continue
        inext = iprev + orderx
        jprev = 0
        j = 0
        for ordery in orders[1]:
            if ordery == 0:
                continue
            jnext = jprev + ordery
            data[i, j] = weighted[iprev:inext, jprev:jnext].sum()
            jprev = jnext
            j += 1
        iprev = inext
        i += 1


class Integrator(FunctionNode):
    """
    The `Integrator` node performs integration (summation)
    of every input within the `weight` and `orders` inputs.

    The `Integrator` has two modes: `1d` and `2d`.
    The `mode` and `precision=dtype` of integration are chosen *automaticly*
    in the type function.

    For `2d` integration arrays should be like `array([...], [...])`
    and must have only the lists with the same length (`orders` too).

    Note that the `Integrator` preallocates temporary buffer.
    For the integration algorithm the `Numba`_ package is used.

    .. note:: `Numba`_ doesn't like arrays of elements with different size,
        so the inputs (including `orders`) must have the elements with the same size.
        This happens due to typing: arrays with elements of different sizes
        have `dtype=object`, but `Numba`_ doesn't like the `object` type
        and works only with numeric and sequence types (including `Numpy`_ types)

    .. _Numba: https://numba.pydata.org
    .. _Numpy: https://numpy.org
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("missing_input_handler", MissingInputAddEach())
        super().__init__(*args, **kwargs)
        self._add_input("weights", positional=False)
        self._add_input("orders", positional=False)

    def _typefunc(self) -> None:
        """
        The function to determine the dtype and shape.
        Checks inputs dimension and, selects an integration algorithm,
        determines dtype and shape for outputs
        """
        check_has_inputs(self)
        check_has_input(self, ("orders", "weights"))
        ndim = len(self.inputs[0].shape)
        if ndim > 2:
            raise TypeFunctionError(
                "The Integrator works only within 1d and 2d mode!", node=self
            )
        check_input_dimension(self, slice(None), ndim)
        check_input_dimension(self, ("orders", "weights"), ndim)
        orders = self.inputs["orders"]
        if not issubdtype(orders.dtype, integer):
            raise TypeFunctionError(
                "The `orders` must be array of integers , but given '{dtype}'!",
                node=self,
                input=orders,
            )
        dtype = self.inputs[0].dtype
        if not issubdtype(dtype, floating):
            raise TypeFunctionError(
                "The Integrator works only within `float` or `double` "
                f"precision, but given '{dtype}'!",
                node=self,
            )
        check_input_dtype(self, slice(None), dtype)
        check_input_dtype(self, ("weights",), dtype)
        self.__integrate = _integrate1d if ndim == 1 else _integrate2d
        if ndim == 1:
            if sum(orders.data) > self.inputs[0].shape[0]:
                raise TypeFunctionError(
                    "Orders must be consistent with inputs lenght, "
                    f"but given {orders.data=} and {self.inputs[0].shape=}!",
                    node=self,
                    input=orders,
                )
        elif any(
            sum(orders.data[i]) > n for i, n in enumerate(self.inputs[0].shape)
        ):
            raise TypeFunctionError(
                "Orders must be consistent with inputs lenght, "
                f"but given {orders.data=} and {self.inputs[0].shape=}!",
                node=self,
                input=orders,
            )
        if ndim == 1:
            newshape = orders.data[orders.data != 0].shape
        else:
            ordersx = orders.data[0]
            ordersy = orders.data[1]
            newshape = (ordersx[ordersx != 0].size, ordersy[ordersy != 0].size)
        for output in self.outputs:
            output._dtype = dtype
            output._shape = newshape

    def post_allocate(self):
        """Allocates the `buffer` within `weights`"""
        weights = self.inputs["weights"]
        self.__buffer = zeros(shape=weights.shape, dtype=weights.dtype)

    def _fcn(self, _, inputs, outputs):
        """
        Integrates inputs within `weights` and `orders` inputs.
        The integration algorithm is selected in `Integrator._typefunc`
        within `Integrator.mode`
        """
        weights = inputs["weights"].data
        orders = inputs["orders"].data
        for input, output in zip(inputs.iter_data(), outputs.iter_data()):
            multiply(input, weights, out=self.__buffer)
            self.__integrate(output, self.__buffer, orders)
        if self.debug:
            return [outputs.iter_data()]
