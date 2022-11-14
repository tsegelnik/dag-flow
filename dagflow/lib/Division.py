from numpy import copyto, result_type

from ..input_extra import MissingInputAddOne
from ..nodes import FunctionNode

class Division(FunctionNode):
    """Division of all the inputs together"""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault(
            "missing_input_handler", MissingInputAddOne(output_fmt="result")
        )
        super().__init__(*args, **kwargs)

    def _fcn(self, _, inputs, outputs):
        out = outputs[0].data
        copyto(out, inputs[0].data.copy())
        if len(inputs) > 1:
            for input in inputs[1:]:
                out /= input.data
        return out

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        self.outputs.result._shape = self.inputs[0].shape
        self.outputs.result._dtype = result_type(
            *tuple(inp.dtype for inp in self.inputs)
        )
        self.logger.debug(
            f"Node '{self.name}': dtype={self.outputs.result.dtype}, "
            f"shape={self.outputs.result.shape}"
        )
