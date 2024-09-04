from numba import njit
from numpy.typing import NDArray

from ..inputhandler import MissingInputAddOne
from ..node import Node
from ..typefunctions import (
    AllPositionals,
    check_has_inputs,
    check_inputs_same_dtype,
    eval_output_dtype,
)


@njit(cache=True)
def _sumsq(data: NDArray, out: NDArray):
    sm = 0.0
    for v in data:
        sm += v * v
    out[:] += sm


class ElSumSq(Node):
    """Sum of the squared of all the inputs"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("missing_input_handler", MissingInputAddOne(output_fmt="result"))
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "Σa²")

    def _fcn(self):
        out = self.outputs["result"].data
        out[:] = 0.0
        for _input in self.inputs.iter_data():
            _sumsq(_input, out)

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        check_has_inputs(self)
        check_inputs_same_dtype(self)
        eval_output_dtype(self, AllPositionals, "result")
        self.outputs[0].dd.shape = (1,)
