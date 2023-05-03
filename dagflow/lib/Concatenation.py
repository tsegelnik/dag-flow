from ..input_extra import MissingInputAddOne
from ..nodes import FunctionNode
from ..typefunctions import (
    check_has_inputs,
    check_input_dimension,
    check_input_dtype,
)


class Concatenation(FunctionNode):
    """
    Creates a node with a single data output which is a concatenated data of the inputs.
    Now supports only 1d arrays.
    """

    __slots__ = ("_offsets",)

    def __init__(self, name, **kwargs):
        kwargs.setdefault(
            "missing_input_handler", MissingInputAddOne(output_fmt="result")
        )
        super().__init__(name, **kwargs)
        self._offsets = []

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        check_has_inputs(self)
        cdtype = self.inputs[0].dd.dtype
        check_input_dtype(self, slice(None), cdtype)
        check_input_dimension(self, slice(None), 1)
        output = self.outputs["result"]
        size = 0
        for input in self.inputs:
            self._offsets.append(size)
            size += input.dd.shape[0]
        output.dd.shape = (size,)
        output.dd.dtype = cdtype

    def _fcn(self, _, inputs, outputs):
        output = outputs["result"]
        data = output.data
        for offset, input in zip(self._offsets, self.inputs):
            size = input.dd.shape[0]
            data[offset : offset + size] = input.data[:]
