from typing import Optional
import numpy as np

from ..nodes import FunctionNode
from ..output import Output
from ..exception import TypeFunctionError

class ViewConcat(FunctionNode):
    """Creates a node with a single data output which is a concatenated memory of the inputs"""

    _output: Output
    _offsets: list[int]
    def __init__(self, name, outname="concat", **kwargs):
        super().__init__(name, **kwargs)
        self._output = self._add_output(outname, allocatable=False)
        self._offsets = []

    def missing_input_handler(self, idx: Optional[int] = None, scope: Optional[int] = None):
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
        cdtype = self.inputs[0].dtype
        for input in self.inputs:
            shape = input.shape
            if len(shape)>1:
                raise TypeFunctionError("ViewConcat supports only 1d inputs", node=self, input=input)
            if input.dtype!=cdtype:
                raise TypeFunctionError("ViewConcat got inconsistent types: {cdtype} and {dtype}", node=self, input=input)

            self._offsets.append(size)
            size+=shape[0]

        output = self.outputs[0]
        output._dtype = cdtype
        output._shape = (size,)
        data = np.zeros(shape=size, dtype=cdtype)
        output._set_data(data, owns_data=True)

        for offset, input in zip(self._offsets, self.inputs):
            size = input.shape[0]
            input.own_data = data[offset:offset+size]



