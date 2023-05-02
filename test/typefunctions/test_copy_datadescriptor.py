#!/usr/bin/env python
from dagflow.graph import Graph
from dagflow.input_extra import MissingInputAddEach, MissingInputAddOne
from dagflow.lib.Array import Array
from dagflow.lib.Dummy import Dummy
from dagflow.typefunctions import (
    AllPositionals,
    copy_from_input_to_output,
    copy_input_shape_to_output,
    eval_output_dtype,
)
from numpy import array
from pytest import mark


@mark.parametrize("dtype", ("i", "d", "float64"))
def test_output_eval_dtype(debug_graph, dtype):
    with Graph(close=False, debug=debug_graph):
        arr1 = Array("arr1", array([1, 2, 3, 4], dtype="i"))
        arr2 = Array("arr2", array([3, 2, 1], dtype=dtype))
        node = Dummy(
            "node",
            missing_input_handler=MissingInputAddOne(output_fmt="result"),
        )
        (arr1, arr2) >> node
    copy_input_shape_to_output(node, 1, "result")
    eval_output_dtype(node, AllPositionals, "result")
    assert node.close()
    assert node.outputs["result"].dd.dtype == dtype


def test_copy_from_input_00(debug_graph):
    with Graph(close=True, debug=debug_graph):
        node = Dummy("node")
    assert (
        copy_from_input_to_output(
            node, slice(None), slice(None), False, False, False, False
        )
        is None
    )


@mark.parametrize("dtype", ("i", "d", "f"))
def test_copy_from_input_01(debug_graph, dtype):
    # TODO: adding axes_nodes check
    with Graph(close=False, debug=debug_graph):
        edges1 = Array("edges1", [0, 1, 2, 3, 4]).outputs["array"]
        edges2 = Array("edges2", [0, 1, 2, 3]).outputs["array"]
        # nodes1 = Array("nodes1", [0.5, 1.5, 2.5, 3.5])
        # nodes2 = Array("nodes2", [0.5, 1.5, 2.5])
        arr1 = Array(
            "arr1", array([1, 2, 3, 4], dtype="i"), edges=edges1
        )  # , nodes=nodes1)
        arr2 = Array(
            "arr2", array([3, 2, 1], dtype=dtype), edges=edges2
        )  # , nodes=nodes2)
        node = Dummy(
            "node",
            missing_input_handler=MissingInputAddEach(),
        )
        arr1 >> node
        arr2 >> node
    copy_from_input_to_output(node, AllPositionals, AllPositionals)
    assert node.close()
    out1 = arr1.outputs["array"].dd
    out2 = arr2.outputs["array"].dd
    assert node.outputs[0].dd.dtype == "i"
    assert node.outputs[0].dd.shape == out1.shape
    assert node.outputs[0].dd.axes_edges == out1.axes_edges
    assert node.outputs[0].dd.axes_nodes == out1.axes_nodes
    assert node.outputs[1].dd.dtype == dtype
    assert node.outputs[1].dd.shape == out2.shape
    assert node.outputs[1].dd.axes_edges == out2.axes_edges
    assert node.outputs[1].dd.axes_nodes == out2.axes_nodes
