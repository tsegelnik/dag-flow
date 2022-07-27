#!/usr/bin/env python

from dagflow.exception import CriticalError
from dagflow.input_extra import MissingInputAddOne
from dagflow.lib import WeightedSum, Array
from dagflow.node import FunctionNode
from numpy import arange, array
from pytest import raises


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

    def check_eval(self):
        super().check_eval()
        if (y := len(self.inputs)) != 3:
            raise CriticalError(f"The node must have 3 inputs, but given {y}!")
        return True

    def _fcn(self, _, inputs, outputs):
        out = outputs[0].data = inputs[0].data.copy()
        for input in inputs[1:]:
            out += input.data


arr = Array("arr", arange(3, dtype="i"))  # [0, 1, 2]


def test_00():
    node = ThreeInputsSum("threesum")
    for _ in range(3):
        # Error while evaluating before len(inputs) == 3
        with raises(CriticalError):
            node.eval()
        arr >> node
    node.close()
    # TODO: if we restrict to close the parent node of outputs,
    #       it is neccessary to close other nodes by hand
    # arr.close()
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
    # TODO: if we restrict to close the parent node of outputs,
    #       it is neccessary to close other nodes by hand
    # arr.close()
    ## ws["weight"].close()
    assert (ws.outputs.result.data == [0, 5, 10]).all()


if __name__ == "__main__":
    test_00()
    test_01()
