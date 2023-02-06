from numba import jit
from numpy import prod, multiply, zeros
from numpy.typing import NDArray

from ..exception import TypeFunctionError
from ..input_extra import MissingInputAddEach
from ..nodes import FunctionNode
from ..typefunctions import check_has_inputs, check_input_dimension


@jit(nopython=True)
def _integrate1d(data: NDArray, weighted: NDArray, orders: NDArray):
    """
    Summing up `weighted` within `orders` and puts the result into `data`.
    The 1-dimensional version.
    """
    iprev = 0
    for i, order in enumerate(orders):
        inext = iprev + order
        data[i] = weighted[iprev:inext].sum()
        iprev = inext


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
    shape = data.shape
    iprev = 0
    for i, orderx in enumerate(orders[0][: shape[0]]):
        inext = iprev + orderx
        jprev = 0
        for j, ordery in enumerate(orders[1][: shape[1]]):
            jnext = jprev + ordery
            data[i, j] = weighted[iprev:inext, jprev:jnext].sum()
            jprev = jnext
        iprev = inext


class Integrator(FunctionNode):
    """
    The `Integrator` node performs integration (summation)
    of every input within the `weight` and `orders` inputs.
    The `precision` of integration can be chosen by `precision` kwarg.

    The `Integrator` has two modes: `1d` and `2d`.
    The mode is chosen by *automaticly* in the type function.

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

    # TODO: design methods for 0d orders

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("missing_input_handler", MissingInputAddEach())
        self._precision = kwargs.pop("precision", "d")
        super().__init__(*args, **kwargs)
        self._add_input("weights", positional=False)
        self._add_input("orders", positional=False)

    @property
    def precision(self):
        """The integration precision"""
        return self._precision

    def _typefunc(self) -> None:
        """
        The function to determine the dtype and shape.
        Checks inputs dimension and, selects an integration algorithm,
        determines dtype and shape for outputs
        """
        check_has_inputs(self)
        ndim = len(self.inputs[0].shape)
        if ndim > 2:
            raise TypeFunctionError(
                "The Integrator works only within 1d and 2d mode!", node=self
            )
        check_input_dimension(self, slice(None), ndim)
        self.__integrate = _integrate1d if ndim == 1 else _integrate2d
        orders = self.inputs["orders"]
        try:
            check_input_dimension(self, ("orders",), ndim)
        except TypeFunctionError as exc:
            try:
                check_input_dimension(self, ("orders",), 0)
            except Exception as exc:
                raise TypeFunctionError(
                    f"The `orders` input must have {ndim=} or 0, but given {len(orders.shape)}",
                    node=self,
                    input=orders,
                ) from exc
        if orders.shape[0] != 1:
            if ndim == 1:
                if sum(orders.data) > self.inputs[0].shape[0]:
                    raise TypeFunctionError(
                        "Orders must be consistent with inputs lenght!",
                        node=self,
                        input=orders,
                    )
            elif any(
                sum(orders.data[i]) > n
                for i, n in enumerate(self.inputs[0].shape)
            ):
                raise TypeFunctionError(
                    "Orders must be consistent with inputs lenght!",
                    node=self,
                    input=orders,
                )
        else:
            # TODO: design checks for ndim - 1 integration
            # for example, `orders.shape(1,5)` for 1d integration of `data.shape(4,5)`
            pass
        for output, input in zip(self.outputs, self.inputs):
            output._dtype = self._precision
            output._shape = input.shape

    def post_allocate(self):
        """Finds the longest input and allocates the `buffer`"""
        self.__buffer = zeros(
            max(prod(input.shape) for input in self.inputs), self._precision
        )

    def _fcn(self, _, inputs, outputs):
        """
        Integrates inputs within `weights` and `orders` inputs.
        The integration algorithm is selected in `Integrator._typefunc`
        within `Integrator.mode`
        """
        weights = inputs["weights"].data
        orders = inputs["orders"].data
        for input, output in zip(inputs.iter_data(), outputs.iter_data()):
            view = self.__buffer.reshape(output.shape)
            multiply(input, weights, out=view)
            self.__integrate(output, view, orders)
        if self.debug:
            return [outputs.iter_data()]
