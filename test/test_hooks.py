#!/usr/bin/env python

from numpy import arange, array, copyto, result_type
from pytest import raises

from dagflow.exception import CriticalError
from dagflow.input_extra import MissingInputAddOne
from dagflow.lib import Array, WeightedSum
from dagflow.node import FunctionNode


class ThreeInputsSum(FunctionNode):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault(
            "missing_input_handler", MissingInputAddOne(output_fmt="result")
        )
        super().__init__(*args, **kwargs)

    def check_input(self, name, iinput=None):
        super().check_input(name, iinput)
        if len(self.inputs) == 3:
            raise CriticalError("The node must have only 3 inputs!")
        return True

    def check_eval(self):
        super().check_eval()
        if (y := len(self.inputs)) != 3:
            raise CriticalError(f"The node must have 3 inputs, but given {y}!")
        return True

    def _fcn(self, _, inputs, outputs):
        out = outputs.result.data
        copyto(out, inputs[0].data.copy())
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


arr = Array("arr", arange(3, dtype="i"))  # [0, 1, 2]


def test_00():
    node = ThreeInputsSum("threesum")
    for _ in range(3):
        # Error while evaluating before len(inputs) == 3
        with raises(CriticalError):
            node.eval()
        arr >> node
    node.close()
    assert (node.outputs.result.data == [0, 3, 6]).all()
    # Error while trying to append fourth input
    with raises(CriticalError):
        arr >> node


def test_01():
    ws = WeightedSum("weightedsum")
    (arr, arr) >> ws
    # Error while eval before setting the weight input
    with raises(CriticalError):
        ws.eval()
    # multiply the first input by 2 and the second one by 3
    Array("weight", (2, 3)) >> ws("weight")
    ws.close()
    assert (ws.outputs.result.data == [0, 5, 10]).all()
