from numba import jit
from numpy import multiply, zeros

from ..exception import InitializationError
from ..input_extra import MissingInputAddEach
from ..nodes import FunctionNode
from ..typefunctions import check_has_inputs


@jit(nopython=True)
def _integrate(data, weighted, orders):
    iprev = 0
    for i, order in enumerate(orders):
        inext = iprev + order
        data[i] = sum(weighted[iprev:inext])
        iprev = inext


class Integrator(FunctionNode):
    """
    Integrator info
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
        self._functions.update(
            {"1d": self._fcn_1d, "2d": self._fcn_2d}
        )

    @property
    def mode(self):
        return self._mode

    @property
    def precision(self):
        return self._precision

    def _typefunc(self) -> None:
        """The function to determine the dtype and shape"""
        check_has_inputs(self)
        self.fcn = self._functions[self._mode]
        for output in self.outputs:
            output._dtype = self._precision
            output._shape = self.inputs["orders"].shape

    def _fcn_1d(self, _, inputs, outputs):
        weights = inputs["weights"].data
        orders = inputs["orders"].data
        weighted = zeros(inputs[0].shape, inputs[0].dtype)
        for input, output in zip(inputs, outputs):
            multiply(input.data, weights, out=weighted)
            _integrate(output.data, weighted, orders)
        if self.debug:
            return [output.data for output in outputs]

    def _fcn_2d(self, _, inputs, outputs):
        pass
