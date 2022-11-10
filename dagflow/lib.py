from numpy import array, copyto, result_type

from .exception import CriticalError, UnconnectedInput
from .input_extra import MissingInputAddOne
from .nodes import FunctionNode, StaticNode
from .output import Output
import numpy as np

class Array(StaticNode):
    """Creates a node with a single data output with predefined array"""

    _data: Output
    def __init__(self, name, arr, outname="array", **kwargs):
        super().__init__(name, **kwargs)
        output = self._add_output(
            outname, allocatable=False, data=array(arr, copy=True)
        )
        self._data = output.data

    def _fcn(self):
        return self.data

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        self.logger.debug(
            f"Node '{self.name}': dtype={self.outputs.array.dtype}, "
            f"shape={self.outputs.array.shape}"
        )

class VariableArray(Array):
    """Creates a node with a single data output with predefined array, enables editing"""

    def set(self, data: np.ndarray) -> bool:
        if self.frozen:
            return False

        self._output.data[:]=data
        return True

class Concatenation(FunctionNode):
    """Creates a node with a single data output from all the inputs data"""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault(
            "missing_input_handler", MissingInputAddOne(output_fmt="result")
        )
        super().__init__(*args, **kwargs)

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        self.outputs.result._shape = tuple(inp.shape for inp in self.inputs)
        self.outputs.result._dtype = result_type(
            *tuple(inp.dtype for inp in self.inputs)
        )
        self.logger.debug(
            f"Node '{self.name}': dtype={self.outputs.result.dtype}, "
            f"shape={self.outputs.result.shape}"
        )

    def _fcn(self, _, inputs, outputs):
        res = outputs.result.data
        res[:] = [inp.data for inp in inputs]
        return res


class Sum(FunctionNode):
    """Sum of all the inputs together"""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault(
            "missing_input_handler", MissingInputAddOne(output_fmt="result")
        )
        super().__init__(*args, **kwargs)

    def _fcn(self, _, inputs, outputs):
        out = outputs.result.data
        copyto(out, inputs[0].data)
        if len(inputs) > 1:
            for input in inputs[1:]:
                out += input.data
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


class Product(FunctionNode):
    """Product of all the inputs together"""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault(
            "missing_input_handler", MissingInputAddOne(output_fmt="result")
        )
        super().__init__(*args, **kwargs)

    def _fcn(self, _, inputs, outputs):
        out = outputs.result.data
        copyto(out, inputs[0].data)
        if len(inputs) > 1:
            for input in inputs[1:]:
                out *= input.data
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


class WeightedSum(FunctionNode):
    """Weighted sum of all the inputs together"""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault(
            "missing_input_handler", MissingInputAddOne(output_fmt="result")
        )
        super().__init__(*args, **kwargs)

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

    @property
    def weight(self):
        for input in self.inputs:
            if input.name in {"weight", "weights"}:
                return input

    def check_input(self, name, iinput=None):
        super().check_input(name, iinput)
        if not self.weight and name not in {"weight", "weights"}:
            raise UnconnectedInput(self, "weight")
        return True

    def check_eval(self):
        super().check_eval()
        if not self.weight:
            raise CriticalError(
                "The `weight` or `weights` input is not setted: "
                "use `WeightedSum.weight = smth` or "
                "`smth >> WeightedSum('weight')`!"
            )
        return True

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
        copyto(out, inputs[0].data * weights[0])
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
