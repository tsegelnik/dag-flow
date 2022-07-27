from itertools import zip_longest

from numpy import asanyarray

from .exception import CriticalError, UnconnectedInput
from .input_extra import MissingInputAddOne
from .node import FunctionNode
from .node_deco import NodeClass
from .tools import IsIterable


class Array(FunctionNode):
    """Creates a note with single data output with predefined array"""

    def __init__(self, name, array, outname="array", **kwargs):
        super().__init__(name, **kwargs)
        self._add_output(outname)
        self.outputs.array.data = asanyarray(array)


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

    def check_input(self, name, iinput=None):
        super().check_input(name, iinput)
        if not self.weight and name not in {"weight", "weights"}:
            raise UnconnectedInput("weight")

    def check_eval(self):
        super().check_eval()
        if not self.weight:
            raise CriticalError(
                "The `weight` or `weights` input is not setted: "
                "use `WeightedSum.weight = smth` or "
                "`smth >> WeightedSum('weight')`!"
            )

    def _fcn(self, _, inputs, outputs):
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
