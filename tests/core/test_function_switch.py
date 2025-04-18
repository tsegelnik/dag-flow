from numpy import arange
from numpy import array
from numpy import copyto
from numpy import result_type

from dagflow.graph import Graph
from dagflow.inputhandler import MissingInputAddOne
from dagflow.lib import Array
from dagflow.node import Node


class SumIntProductFloatElseNothing(Node):
    def __init__(self, name, **kwargs):
        kwargs.setdefault("missing_input_handler", MissingInputAddOne(output_fmt="result"))
        super().__init__(name, **kwargs)
        self._functions.update({"int": self._fcn_int, "float": self._fcn_float})

    def _fcn(self):
        return self.outputs[0].data

    def _fcn_int(self):
        out = self.outputs[0].data
        copyto(out, self.inputs[0].data.copy())
        if len(self.inputs) > 1:
            for _input in self.inputs[1:]:
                out += _input.data
        return out

    def _fcn_float(self):
        out = self.outputs[0].data
        copyto(out, self.inputs[0].data.copy())
        if len(self.inputs) > 1:
            for _input in self.inputs[1:]:
                out *= _input.data
        return out

    def _typefunc(self) -> bool:
        if self.inputs[0].dd.dtype == "i":
            self.fcn = self._functions.get("int")
        elif self.inputs[0].dd.dtype == "d":
            self.fcn = self._functions.get("float")
        self.outputs["result"].dd.shape = self.inputs[0].dd.shape
        self.outputs["result"].dd.dtype = result_type(*tuple(inp.dd.dtype for inp in self.inputs))
        self.logger.debug(
            f"Node '{self.name}': dtype={self.outputs['result'].dd.dtype}, "
            f"shape={self.outputs['result'].dd.shape}, function={self.fcn.__name__}"
        )
        return True


def test_00(debug_graph):
    with Graph(debug=debug_graph, close_on_exit=True):
        arr = Array("arr", array(("1", "2", "3")))
        node = SumIntProductFloatElseNothing("node")
        (arr, arr) >> node
    assert (node.outputs["result"].data == ["", "", ""]).all()


def test_01(debug_graph):
    with Graph(debug=debug_graph, close_on_exit=True):
        arr = Array("arr", arange(3, dtype="i"))  # [0, 1, 2]
        node = SumIntProductFloatElseNothing("node")
        (arr, arr) >> node
    assert (node.outputs["result"].data == [0, 2, 4]).all()


def test_02(debug_graph):
    with Graph(debug=debug_graph, close_on_exit=True):
        arr = Array("arr", arange(3, dtype="d"))  # [0, 1, 2]
        node = SumIntProductFloatElseNothing("node")
        (arr, arr) >> node
    assert (node.outputs["result"].data == [0, 1, 4]).all()
