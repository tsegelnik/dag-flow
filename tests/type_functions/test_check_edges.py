from numpy import array
from pytest import mark

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.input_handler import MissingInputAddOne
from dagflow.lib.common import Array
from dagflow.lib.common import Dummy
from dagflow.type_functions import (
    AllPositionals,
    check_array_edges_consistency,
    check_edges_type,
    check_input_edges_dim,
    check_input_edges_equivalence,
    copy_from_input_to_output,
    copy_input_edges_to_output,
)


@mark.parametrize(
    "data,edgesdata",
    (
        ([1], [1, 2]),
        ([1, 2], [1, 2, 3]),
        ([1, 2, 3], [1, 2, 3, 4]),
    ),
)
def test_edges_00(testname, debug_graph, data, edgesdata):
    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        edges = Array("edges", edgesdata).outputs["array"]
        arr1 = Array("arr1", array(data), edges=edges)
        arr2 = Array("arr2", 2 * array(data), edges=edges)
        node = Dummy(
            "node",
            missing_input_handler=MissingInputAddOne(output_fmt="result"),
        )
        arr1 >> node
        arr2 >> node
        copy_from_input_to_output(node, 0, "result")
        check_input_edges_dim(node, AllPositionals)
        check_input_edges_equivalence(node, AllPositionals)
        check_edges_type(node)
        check_array_edges_consistency(node, "result")
        copy_input_edges_to_output(node, 0, "result")
    savegraph(graph, f"output/{testname}.png")


@mark.parametrize(
    "data,edgesdataX,edgesdataY",
    (
        ([[1], [1]], [1, 2, 3], [1, 2]),
        ([[1, 2], [1, 2]], [1, 2, 3], [1, 2, 3]),
        ([[1, 2, 3], [1, 2, 3]], [1, 2, 3], [1, 2, 3, 4]),
        ([[1, 2], [1, 2], [1, 2]], [1, 2, 3, 4], [1, 2, 3]),
        ([[1, 2, 3], [1, 2, 3], [1, 2, 3]], [1, 2, 3, 4], [1, 2, 3, 4]),
    ),
)
def test_edges_01(testname, debug_graph, data, edgesdataX, edgesdataY):
    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        edgesX = Array("edgesX", edgesdataX).outputs["array"]
        edgesY = Array("edgesY", edgesdataY).outputs["array"]
        edges = [edgesX, edgesY]
        arr1 = Array("arr1", array(data), edges=edges)
        arr2 = Array("arr2", 2 * array(data), edges=edges)
        node = Dummy(
            "node",
            missing_input_handler=MissingInputAddOne(output_fmt="result"),
        )
        arr1 >> node
        arr2 >> node
        copy_from_input_to_output(node, 0, "result")
        check_input_edges_dim(node, AllPositionals)
        check_input_edges_equivalence(node, AllPositionals)
        check_edges_type(node)
        check_array_edges_consistency(node, "result")
        copy_input_edges_to_output(node, 0, "result")
    savegraph(graph, f"output/{testname}.png")
