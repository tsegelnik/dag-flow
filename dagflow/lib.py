from itertools import zip_longest

from numpy import array, copyto, result_type

from .exception import CriticalError, UnconnectedInput
from .input_extra import MissingInputAddOne
from .node import FunctionNode, StaticNode
#from .node_deco import NodeClass
from .tools import IsIterable


class Array(StaticNode):
    """Creates a note with single data output with predefined array"""

    def __init__(self, name, arr, outname="array", **kwargs):
        super().__init__(name, **kwargs)
        self._add_output(outname, allocatable=False, data=array(arr, copy=True))

    def _shapefunc(self, node) -> None:
        """A output takes this function to determine the shape"""
        return node.outputs.array.shape

    def _typefunc(self, node) -> None:
        """A output takes this function to determine the dtype"""
        return node.outputs.array.dtype


class Sum(FunctionNode):
    """Sum of all the inputs together"""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault(
            "missing_input_handler", MissingInputAddOne(output_fmt="result")
        )
        super().__init__(*args, **kwargs)

    def _fcn(self, _, inputs, outputs):
        out = outputs.result.data
        copyto(out, inputs[0].data.copy())
        if len(inputs) > 1:
            for input in inputs[1:]:
                out += input.data

    def _shapefunc(self, node) -> None:
        """A output takes this function to determine the shape"""
        return node.inputs[0].data.shape

    def _typefunc(self, node) -> None:
        """A output takes this function to determine the dtype"""
        return result_type(*tuple(inp.dtype for inp in node.inputs))


class Product(FunctionNode):
    """Product of all the inputs together"""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault(
            "missing_input_handler", MissingInputAddOne(output_fmt="result")
        )
        super().__init__(*args, **kwargs)

    def _fcn(self, _, inputs, outputs):
        out = outputs.result.data
        copyto(out, inputs[0].data.copy())
        if len(inputs) > 1:
            for input in inputs[1:]:
                out *= input.data

    def _shapefunc(self, node) -> None:
        """A output takes this function to determine the shape"""
        return node.inputs[0].data.shape

    def _typefunc(self, node) -> None:
        """A output takes this function to determine the dtype"""
        return result_type(*tuple(inp.dtype for inp in node.inputs))


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

    def _shapefunc(self, node) -> None:
        """A output takes this function to determine the shape"""
        return node.inputs[0].data.shape

    def _typefunc(self, node) -> None:
        """A output takes this function to determine the dtype"""
        return result_type(*tuple(inp.dtype for inp in node.inputs))


class WeightedSum(FunctionNode):
    """Weighted sum of all the inputs together"""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault(
            "missing_input_handler", MissingInputAddOne(output_fmt="result")
        )
        super().__init__(*args, **kwargs)

    def _shapefunc(self, node) -> None:
        """A output takes this function to determine the shape"""
        return node.inputs[0].data.shape

    def _typefunc(self, node) -> None:
        """A output takes this function to determine the dtype"""
        return result_type(*tuple(inp.dtype for inp in node.inputs))

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
        return self.__fcn_iterable(self.weight.data, inputs, outputs)

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
        copyto(out, inputs[0].data.copy() * weights[0])
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
