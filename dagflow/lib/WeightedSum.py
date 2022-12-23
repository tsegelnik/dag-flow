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
        self._functions.update(
            {"number": self.__fcn_number, "iterable": self.__fcn_iterable}
        )

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        weight = self.inputs.get("weight")
        if weight is None:
            raise TypeFunctionError("Cannot use WeightedSum without 'weight'!")
        input = next(
            (inp for inp in self.inputs if inp.name != "weight"), None
        )
        if input is None:
            raise TypeFunctionError(
                "Cannot use WeightedSum with zero arguments!"
            )
        if len(weight.parent_node._data) == 0:
            raise TypeFunctionError("Cannot use WeightedSum with empty 'weight'!")
        self.fcn = self._functions.get(
            "number" if len(weight.parent_node._data) == 1 else "iterable"
        )
        self.outputs["result"]._shape = input.shape
        self.outputs["result"]._dtype = result_type(
            *tuple(inp.dtype for inp in self.inputs)
        )

    def _fcn(self, _, inputs, outputs):
        inputs = tuple(input for input in inputs if input.name != "weight")
        return self.fcn(_, inputs, outputs)

    def __fcn_number(self, _, inputs, outputs):
        out = outputs[0].data
        weight = self.inputs["weight"].data
        copyto(out, inputs[0].data.copy())
        if len(inputs) > 1:
            for input in inputs[1:]:
                out += input.data
        return out * weight

    def __fcn_iterable(self, _, inputs, outputs):
        out = outputs[0].data
        weights = self.inputs["weight"].data
        copyto(out, inputs[0].data * weights[0])
        if len(inputs) > 1:
            for input, weight in zip(inputs[1:], weights[1:]):
                print(f"{weight=}, {input=}")
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
