from numpy import add, empty, square
from numpy.typing import NDArray

from ..typefunctions import (
    check_has_inputs,
    eval_output_dtype,
    copy_input_shape_to_outputs,
    check_inputs_equivalence,
    AllPositionals
)
from .ManyToOneNode import ManyToOneNode

class SumSq(ManyToOneNode):
    """Sum of the squares of all the inputs"""

    __slots__ = ("_buffer",)
    _buffer: NDArray

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "Σ()²")

    def _fcn(self) -> NDArray:
        out = self.outputs["result"].data
        square(self.inputs[0].data, out=out)
        if len(self.inputs) > 1:
            for _input in self.inputs[1:]:
                square(_input.data, out=self._buffer)
                add(self._buffer, out, out=out)
        return out

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        check_has_inputs(self)
        copy_input_shape_to_outputs(self, 0, "result")
        check_inputs_equivalence(self)
        eval_output_dtype(self, AllPositionals, "result")

    def _post_allocate(self) -> None:
        inpdd = self.inputs[0].dd
        self._buffer = empty(shape=inpdd.shape, dtype=inpdd.dtype)
