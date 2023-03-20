from typing import List, Optional

from numpy import zeros

from ..nodes import FunctionNode
from ..output import Output
from ..typefunctions import check_input_dimension, check_input_dtype


class ViewConcat(FunctionNode):
    """Creates a node with a single data output which is a concatenated memory of the inputs"""

    _output: Output
    _offsets: List[int]

    def __init__(self, name, outname="concat", **kwargs):
        super().__init__(name, **kwargs)
        self._output = self._add_output(
            outname, allocatable=False, forbid_reallocation=True
        )
        self._offsets = []

    def missing_input_handler(
        self, idx: Optional[int] = None, scope: Optional[int] = None
    ):
        icount = len(self.inputs)
        idx = idx if idx is not None else icount
        iname = "input_{:02d}".format(idx)

        kwargs = {"child_output": self._output}
        return self._add_input(iname, allocatable=True, **kwargs)

    def _fcn(self, _, inputs, outputs):
        self.inputs.touch()
        return self._output._data

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        size = 0
        self._offsets = []
        cdtype = self.inputs[0].dd.dtype
        check_input_dtype(self, slice(None), cdtype)
        check_input_dimension(self, slice(None), 1)
        for input in self.inputs:
            self._offsets.append(size)
            size += input.dd.shape[0]

        output = self.outputs[0]
        output.dd.dtype = cdtype
        output.dd.shape = (size,)
        data = zeros(shape=size, dtype=cdtype)
        output._set_data(data, owns_buffer=True)

        for offset, input in zip(self._offsets, self.inputs):
            size = input.dd.shape[0]
            idata = data[offset : offset + size]
            input.set_own_data(idata, owns_buffer=False)
