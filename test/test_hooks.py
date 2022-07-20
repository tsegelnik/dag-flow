#!/usr/bin/env python

from dagflow.exception import CriticalError
from dagflow.input_extra import MissingInputAddOne
from dagflow.lib import WeightedSum, makeArray
from dagflow.node import FunctionNode
from numpy import arange, array
from pytest import raises


class ThreeInputsSum(FunctionNode):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault(
            "missing_input_handler", MissingInputAddOne(output_fmt="result")
        )
        super().__init__(*args, **kwargs)

    def _check_input(self, name, iinput=None):
        if len(self.inputs) == 3:
            raise CriticalError("The node must have only 3 inputs!")

    def _check_eval(self):
        if (y := len(self.inputs)) != 3:
            raise CriticalError(f"The node must have 3 inputs, but given {y}!")

    def _fcn(self, _, inputs, outputs):
        out = outputs[0].data = inputs[0].data.copy()
        for input in inputs[1:]:
            out += input.data


arr = makeArray(arange(3, dtype="i"))("arr")  # [0, 1, 2]


def test_00():
    node = ThreeInputsSum("threesum")
    for _ in range(3):
        # Error while evaluating before len(inputs) == 3
        with raises(CriticalError):
            node.eval()
        arr >> node
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
    makeArray((2, 3))("weight") >> ws("weight")
    assert (ws.outputs.result.data == [0, 5, 10]).all()


if __name__ == "__main__":
    test_00()
    test_01()
