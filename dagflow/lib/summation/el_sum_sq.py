from numba import njit
from numpy.typing import NDArray

from ...core.input_handler import MissingInputAddOne
from ...core.node import Node
from ...core.type_functions import (
    AllPositionals,
    check_node_has_inputs,
    check_inputs_have_same_dtype,
    evaluate_dtype_of_outputs,
)


@njit(cache=True)
def _sumsq(data: NDArray, out: NDArray):
    sm = 0.0
    for v in data:
        sm += v * v
    out[0] += sm


class ElSumSq(Node):
    """Sum of the squared of all the inputs"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("missing_input_handler", MissingInputAddOne(output_fmt="result"))
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "Σa²")

    def _function(self):
        out = self.outputs["result"].data
        out[0] = 0.0
        for _input in self.inputs.iter_data():
            _sumsq(_input, out)

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        check_node_has_inputs(self)
        check_inputs_have_same_dtype(self)
        evaluate_dtype_of_outputs(self, AllPositionals, "result")
        self.outputs[0].dd.shape = (1,)
