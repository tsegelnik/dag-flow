from numpy import copyto, result_type

from ..exception import TypeFunctionError
from ..input_extra import MissingInputAddOne
from ..nodes import FunctionNode


class WeightedSum(FunctionNode):
    """Weighted sum of all the inputs together"""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault(
            "missing_input_handler", MissingInputAddOne(output_fmt="result")
        )
        super().__init__(*args, **kwargs)
        self._add_input("weight", positional=False)
        self._functions.update(
            {"number": self._fcn_number, "iterable": self._fcn_iterable}
        )

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        weight = self.inputs["weight"]
        shape = weight.shape[0]
        leninp = len(self.inputs)
        if leninp == 0:
            raise TypeFunctionError(
                "Cannot use WeightedSum with zero inputs!"
            )
        if shape == 0:
            raise TypeFunctionError(
                "Cannot use WeightedSum with empty 'weight'!"
            )
        elif shape == 1:
            self.fcn = self._functions["number"]
        elif shape == leninp:
            self.fcn = self._functions["iterable"]
        else:
            raise TypeFunctionError(
                f"The number of weights (={shape}) must coinside "
                f"with the number of inputs (={leninp})!"
            )
        self.outputs["result"]._shape = self.inputs[0].shape
        self.outputs["result"]._dtype = result_type(
            *tuple(inp.dtype for inp in self.inputs)
        )

    def _fcn_number(self, _, inputs, outputs):
        """
        The function for one weight for all inputs:
        `len(weight) == 1`
        """
        out = outputs[0].data
        weight = self.inputs["weight"].data
        copyto(out, inputs[0].data.copy())
        if len(inputs) > 1:
            for input in inputs[1:]:
                out += input.data
        out *= weight
        return out

    def _fcn_iterable(self, _, inputs, outputs):
        """
        The function for one weight for every input:
        `len(weight) == len(inputs)`
        """
        out = outputs[0].data
        weights = self.inputs["weight"].data
        copyto(out, inputs[0].data * weights[0])
        if len(inputs) > 1:
            for input, weight in zip(inputs[1:], weights[1:]):
                out += input.data * weight
        return out
