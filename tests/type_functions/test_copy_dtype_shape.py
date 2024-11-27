from numpy import array
from pytest import mark

from dagflow.core.graph import Graph
from dagflow.plot.graphviz import savegraph
from dagflow.core.input_handler import MissingInputAddEach, MissingInputAddOne
from dagflow.lib.common import Array
from dagflow.lib.common import Dummy
from dagflow.core.type_functions import (
    AllPositionals,
    copy_from_inputs_to_outputs,
    copy_dtype_from_inputs_to_outputs,
    copy_shape_from_inputs_to_outputs,
    evaluate_dtype_of_outputs,
)


@mark.parametrize(
    "data,dtype",
    (
        ([1, 2, 3], "i"),
        ([[1, 2], [3, 4]], "d"),
        ([[[1, 2], [3, 4], [5, 6]]], "float64"),
    ),
)
def test_copy_input_dtype_and_shape(testname, debug_graph, data, dtype):
    with Graph(close_on_exit=False, debug=debug_graph) as graph:
        arrdata = array(data, dtype=dtype)
        arr1 = Array("arr1", arrdata)
        node = Dummy(
            "node",
            missing_input_handler=MissingInputAddOne(output_fmt="result"),
        )
        arr1 >> node
    copy_shape_from_inputs_to_outputs(node, 0, "result")
    copy_dtype_from_inputs_to_outputs(node, 0, "result")
    assert node.close()
    assert node.outputs["result"].dd.dtype == arrdata.dtype
    assert node.outputs["result"].dd.shape == arrdata.shape
    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("dtype", ("i", "d", "float64"))
def test_output_eval_dtype(testname, debug_graph, dtype):
    with Graph(close_on_exit=False, debug=debug_graph) as graph:
        arr1 = Array("arr1", array([1, 2, 3, 4], dtype="i"))
        arr2 = Array("arr2", array([3, 2, 1], dtype=dtype))
        node = Dummy(
            "node",
            missing_input_handler=MissingInputAddOne(output_fmt="result"),
        )
        (arr1, arr2) >> node
    copy_shape_from_inputs_to_outputs(node, 1, "result")
    evaluate_dtype_of_outputs(node, AllPositionals, "result")
    assert node.close()
    assert node.outputs["result"].dd.dtype == dtype
    savegraph(graph, f"output/{testname}.png")


def test_copy_from_input_00(testname, debug_graph):
    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        node = Dummy("node")
    assert (
        copy_from_inputs_to_outputs(
            node,
            slice(None),
            slice(None),
            dtype=False,
            shape=False,
            edges=False,
            meshes=False,
        )
        is None
    )
    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("dtype", ("i", "d", "f"))
def test_copy_from_input_01(testname, debug_graph, dtype):
    # TODO: adding axes_meshes check
    with Graph(close_on_exit=False, debug=debug_graph) as graph:
        edges1 = Array("edges1", [0, 1, 2, 3, 4]).outputs["array"]
        edges2 = Array("edges2", [0, 1, 2, 3]).outputs["array"]
        # nodes1 = Array("nodes1", [0.5, 1.5, 2.5, 3.5])
        # nodes2 = Array("nodes2", [0.5, 1.5, 2.5])
        arr1 = Array("arr1", array([1, 2, 3, 4], dtype="i"), edges=edges1)  # , nodes=nodes1)
        arr2 = Array("arr2", array([3, 2, 1], dtype=dtype), edges=edges2)  # , nodes=nodes2)
        node = Dummy(
            "node",
            missing_input_handler=MissingInputAddEach(),
        )
        arr1 >> node
        arr2 >> node
    copy_from_inputs_to_outputs(node, AllPositionals, AllPositionals)
    assert node.close()
    out1 = arr1.outputs["array"].dd
    out2 = arr2.outputs["array"].dd
    assert node.outputs[0].dd.dtype == "i"
    assert node.outputs[0].dd.shape == out1.shape
    assert node.outputs[0].dd.axes_edges == out1.axes_edges
    assert node.outputs[0].dd.axes_meshes == out1.axes_meshes
    assert node.outputs[1].dd.dtype == dtype
    assert node.outputs[1].dd.shape == out2.shape
    assert node.outputs[1].dd.axes_edges == out2.axes_edges
    assert node.outputs[1].dd.axes_meshes == out2.axes_meshes
    savegraph(graph, f"output/{testname}.png")
