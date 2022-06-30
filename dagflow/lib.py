from itertools import zip_longest

from .input_extra import MissingInputAddOne
from .node import FunctionNode
from .node_deco import NodeClass
from .tools import IsIterable


def makeArray(arr):
    @NodeClass(output="array")
    def cls(node, inputs, outputs):
        """Creates a node with single data output with predefined array"""
        outputs[0].data = arr

    return cls


class Sum(FunctionNode):
    """Sum of all the inputs together"""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault(
            "missing_input_handler", MissingInputAddOne(output_fmt="result")
        )
        super().__init__(*args, **kwargs)

    def _fcn(self, _, inputs, outputs):
        out = outputs[0].data = inputs[0].data.copy()
        if len(inputs) > 1:
            for input in inputs[1:]:
                out += input.data


class Product(FunctionNode):
    """Product of all the inputs together"""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault(
            "missing_input_handler", MissingInputAddOne(output_fmt="result")
        )
        super().__init__(*args, **kwargs)

    def _fcn(self, _, inputs, outputs):
        out = outputs[0].data = inputs[0].data.copy()
        if len(inputs) > 1:
            for input in inputs[1:]:
                out *= input.data


class Division(FunctionNode):
    """Division of all the inputs together"""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault(
            "missing_input_handler", MissingInputAddOne(output_fmt="result")
        )
        super().__init__(*args, **kwargs)

    def _fcn(self, _, inputs, outputs):
        out = outputs[0].data = inputs[0].data.copy()
        if len(inputs) > 1:
            for input in inputs[1:]:
                out /= input.data


class WeightedSum(FunctionNode):
    """Weighted sum of all the inputs together"""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault(
            "missing_input_handler", MissingInputAddOne(output_fmt="result")
        )
        super().__init__(*args, **kwargs)

    @property
    def weight(self):
        for input in self.inputs:
            if input.name in {"weight", "weights"}:
                return input

    def _fcn(self, _, inputs, outputs):
        if not self.weight:
            # TODO: do we need exception or warning?
            raise RuntimeError(
                "The `weight` or `weights` input is not setted: "
                "use `WeightedSum.weight = smth` or "
                "`smth >> WeightedSum('weight')`!"
            )
        inputs = tuple(
            input
            for input in inputs
            if input.name not in {"weight", "weights"}
        )
        if self.weight.datatype in (int, float):
            return self.__fcn_number(self.weight.data, inputs, outputs)
        elif IsIterable(self.weight.data) and len(self.weight.data) != 0:
            return self.__fcn_iterable(self.weight.data, inputs, outputs)
        raise RuntimeError(
            "There is no implementation of the WeightedSum for "
            f"{self.weight.data, self.weight.datatype}!"
        )

    def __fcn_number(self, weight, inputs, outputs):
        out = outputs[0].data = inputs[0].data.copy()
        if len(inputs) > 1:
            for input in inputs[1:]:
                out += input.data
        return out * weight

    def __fcn_iterable(self, weights, inputs, outputs):
        if len(weights) == 1:
            return self.__fcn_number(weights[0], inputs, outputs)
        out = outputs[0].data = inputs[0].data.copy()
        out *= weights[0]
        if len(inputs) > 1:
            for input, weight in zip(inputs[1:], weights[1:]):
                if input is None:
                    # Should we raise an exception or warning,
                    # if len(weights) > len(inputs)?
                    raise RuntimeError(
                        f"The {len(weights)=} > {len(inputs)=}!"
                    )
                if weight is None:
                    # Should we raise an exception or warning,
                    # if len(weights) < len(inputs)?
                    raise RuntimeError(
                        f"The {len(inputs)=} > {len(weights)=}!"
                    )
                out += input.data * weight

        return out
