#!/usr/bin/env python
from numpy import arange
from numpy import copyto
from pytest import raises

from dagflow.exception import CriticalError
from dagflow.exception import ReconnectionError
from dagflow.exception import UnclosedGraphError
from dagflow.graph import Graph
from dagflow.lib import Array
from dagflow.lib import ManyToOneNode
from dagflow.lib import WeightedSum


class ThreeInputsSum(ManyToOneNode):
    def _fcn(self):
        out = self.outputs["result"].data
        copyto(out, self.inputs[0].data.copy())
        for _input in self.inputs[1:3]:
            out += _input.data
        return out

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        super()._typefunc()
        if (y := len(self.inputs)) != 3:
            raise CriticalError(
                f"The node must have only 3 inputs, but given {y}: {self.inputs}!"
            )
        self.logger.debug(
            f"Node '{self.name}': dtype={self.outputs['result'].dd.dtype}, "
            f"shape={self.outputs['result'].dd.shape}"
        )


def test_00(debug_graph):
    with Graph(debug=debug_graph, close_on_exit=True):
        arr = Array("arr", arange(3, dtype="i"))  # [0, 1, 2]
        node = ThreeInputsSum("threesum")
        for _ in range(3):
            # Error while evaluating before len(inputs) == 3
            with raises(UnclosedGraphError):
                node.eval()
            arr >> node
    assert (node.outputs["result"].data == [0, 3, 6]).all()


def test_01(debug_graph):
    with Graph(debug=debug_graph, close_on_exit=True):
        arr = Array("arr", arange(3, dtype="i"))  # [0, 1, 2]
        ws = WeightedSum("weightedsum")
        (arr, arr) >> ws
        # Error while eval before setting the weight input
        with raises(UnclosedGraphError):
            ws.eval()
        # multiply the first input by 2 and the second one by 3
        Array("weight", (2, 3)) >> ws("weight")
    with raises(ReconnectionError):
        Array("weight", (2, 3)) >> ws("weight")
    assert (ws.outputs["result"].data == [0, 5, 10]).all()
