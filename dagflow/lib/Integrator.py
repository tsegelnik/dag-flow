from numba import jit
from numpy import multiply, zeros

from ..exception import InitializationError
from ..input_extra import MissingInputAddEach
from ..nodes import FunctionNode
from ..typefunctions import check_has_inputs, check_input_dimension


@jit(nopython=True)
def _integrate1d(data, weighted, orders):
    iprev = 0
    for i, order in enumerate(orders):
        inext = iprev + order
        data[i] = weighted[iprev:inext].sum()
        iprev = inext


@jit(nopython=True)
def _integrate2d(data, weighted, orders):
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
    The mode is chosen by `mode` kwarg.

    For `2d` integration arrays should be like `array([...], [...])`
    and must have only the lists with the same length (`orders` too).

    Note that the `Integrator` preallocates temporary buffer.

    .. note:: `Numba`_ doesn't like , so the inputs (including `orders`)
        must have the elements with the same size.
        This happens due to typing: arrays with elements of different sizes
        have `dtype=object`, but `Numba`_ doesn't like the `object` type
        and works only with numeric types (including `Numpy`_ types)

    .. _Numba: https://numba.pydata.org
    .. _Numpy: https://numpy.org
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("missing_input_handler", MissingInputAddEach())
        if (mode := kwargs.pop("mode", None)) is None:
            self._mode = "1d"
        elif mode not in {"1d", "2d"}:
            raise InitializationError(
                f"Mode must be '1d' or '2d', but given {mode}", node=self
            )
        else:
            self._mode = mode
        self._precision = kwargs.pop("precision", "d")
        super().__init__(*args, **kwargs)
        self._add_input("weights", positional=False)
        self._add_input("orders", positional=False)
        self._functions.update({"1d": self._fcn_1d, "2d": self._fcn_2d})

    @property
    def mode(self):
        return self._mode

    @property
    def precision(self):
        return self._precision

    def _typefunc(self) -> None:
        """The function to determine the dtype and shape"""
        check_has_inputs(self)
        check_input_dimension(
            self, slice(None), 1 if self._mode == "1d" else 2
        )
        self.fcn = self._functions[self._mode]
        for output, input in zip(self.outputs, self.inputs):
            output._dtype = self._precision
            output._shape = input.shape

    def post_allocate(self):
        """Find the longest input and allocate the `buffer`"""
        shape = self.inputs[0].shape
        dtype = self.inputs[0].dtype
        if len(self.inputs) > 1:
            for input in self.inputs[1:]:
                for i, j in zip(input.shape, shape):
                    if i > j:
                        shape = input.shape
                        dtype = input.dtype
                        break
                    elif i < j:
                        break
        self.__buffer = zeros(shape, dtype)

    def _fcn_1d(self, _, inputs, outputs):
        weights = inputs["weights"].data
        orders = inputs["orders"].data
        for input, output in zip(inputs.iter_data(), outputs.iter_data()):
            multiply(input, weights, out=self.__buffer)
            _integrate1d(output, self.__buffer, orders)
        if self.debug:
            return [outputs.iter_data()]

    def _fcn_2d(self, _, inputs, outputs):
        weights = inputs["weights"].data
        orders = inputs["orders"].data
        for input, output in zip(inputs.iter_data(), outputs.iter_data()):
            multiply(input, weights, out=self.__buffer)
            _integrate2d(output, self.__buffer, orders)
        if self.debug:
            return [outputs.iter_data()]
