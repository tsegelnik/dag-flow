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

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        if self.inputs.get("weight") is None:
            raise TypeFunctionError("Cannot use WeightedSum without 'weight'!")
        input = next(
            (inp for inp in self.inputs if inp.name != "weight"), None
        )
        if input is None:
            raise TypeFunctionError("Cannot use WeightedSum with zero arguments!")
        self.outputs["result"]._shape = input.shape
        self.outputs["result"]._dtype = result_type(
            *tuple(inp.dtype for inp in self.inputs)
        )
        self.logger.debug(
            f"Node '{self.name}': dtype={self.outputs['result'].dtype}, "
            f"shape={self.outputs['result'].shape}"
        )

    def _fcn(self, _, inputs, outputs):
        inputs = tuple(input for input in inputs if input.name != "weight")
        return self.__fcn_iterable(self.inputs["weight"].data, inputs, outputs)

    def __fcn_number(self, weight, inputs, outputs):
        out = outputs[0].data
        copyto(out, inputs[0].data.copy())
        if len(inputs) > 1:
            for input in inputs[1:]:
                out += input.data
        return out * weight

    def __fcn_iterable(self, weights, inputs, outputs):
        if len(weights) == 1:
            return self.__fcn_number(weights[0], inputs, outputs)
        out = outputs[0].data
        copyto(out, inputs[0].data * weights[0])
        if len(inputs) > 1:
            for input, weight in zip(inputs[1:], weights[1:]):
                if input is None:
                    # TODO: Should we raise an exception or warning,
                    # if len(weights) > len(inputs)?
                    raise RuntimeError(
                        f"The {len(weights)=} > {len(inputs)=}!"
                    )
                if weight is None:
                    # TODO: Should we raise an exception or warning,
                    # if len(weights) < len(inputs)?
                    raise RuntimeError(
                        f"The {len(inputs)=} > {len(weights)=}!"
                    )
                out += input.data * weight
        return out
