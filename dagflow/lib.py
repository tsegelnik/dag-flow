from .input_extra import MissingInputAddOne
from .node_deco import NodeClass
from .node import FunctionNode


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
