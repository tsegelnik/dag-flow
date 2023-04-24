#!/usr/bin/env python

from dagflow.graph import Graph
from dagflow.input_extra import MissingInputAddOne
from dagflow.lib import Array
from dagflow.nodes import FunctionNode
from numpy import arange, array, copyto, result_type


class SumIntProductFloatElseNothing(FunctionNode):
    def __init__(self, name, **kwargs):
        kwargs.setdefault(
            "missing_input_handler", MissingInputAddOne(output_fmt="result")
        )
        super().__init__(name, **kwargs)
        self._functions.update(
            {"int": self._fcn_int, "float": self._fcn_float}
        )

    def _fcn(self, _, inputs, outputs):
        return outputs[0].data

    def _fcn_int(self, _, inputs, outputs):
        out = outputs[0].data
        copyto(out, inputs[0].data.copy())
        if len(inputs) > 1:
            for input in inputs[1:]:
                out += input.data
        return out

    def _fcn_float(self, _, inputs, outputs):
        out = outputs[0].data
        copyto(out, inputs[0].data.copy())
        if len(inputs) > 1:
            for input in inputs[1:]:
                out *= input.data
        return out

    def _typefunc(self) -> bool:
        if self.inputs[0].dd.dtype == "i":
            self.fcn = self._functions.get("int")
        elif self.inputs[0].dd.dtype == "d":
            self.fcn = self._functions.get("float")
        self.outputs["result"].dd.shape = self.inputs[0].dd.shape
        self.outputs["result"].dd.dtype = result_type(
            *tuple(inp.dd.dtype for inp in self.inputs)
        )
        self.logger.debug(
            f"Node '{self.name}': dtype={self.outputs['result'].dd.dtype}, "
            f"shape={self.outputs['result'].dd.shape}, function={self.fcn.__name__}"
        )
        return True


def test_00(debug_graph):
    with Graph(debug=debug_graph, close=True):
        arr = Array("arr", array(("1", "2", "3")))
        node = SumIntProductFloatElseNothing("node")
        (arr, arr) >> node
    assert (node.outputs["result"].data == ["", "", ""]).all()


def test_01(debug_graph):
    with Graph(debug=debug_graph, close=True):
        arr = Array("arr", arange(3, dtype="i"))  # [0, 1, 2]
        node = SumIntProductFloatElseNothing("node")
        (arr, arr) >> node
    assert (node.outputs["result"].data == [0, 2, 4]).all()


def test_02(debug_graph):
    with Graph(debug=debug_graph, close=True):
        arr = Array("arr", arange(3, dtype="d"))  # [0, 1, 2]
        node = SumIntProductFloatElseNothing("node")
        (arr, arr) >> node
    assert (node.outputs["result"].data == [0, 1, 4]).all()
