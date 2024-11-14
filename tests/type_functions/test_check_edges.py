from numpy import array
from pytest import mark

from dagflow.core.graph import Graph
from dagflow.plot.graphviz import savegraph
from dagflow.core.input_handler import MissingInputAddOne
from dagflow.lib.common import Array
from dagflow.lib.common import Dummy
from dagflow.core.type_functions import (
    AllPositionals,
    check_edges_consistency_with_array,
    check_dtype_of_edges,
    check_edges_dimension_of_inputs,
    check_edges_equivalence_of_inputs,
    copy_from_inputs_to_outputs,
    copy_edges_from_inputs_to_outputs,
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
        copy_from_inputs_to_outputs(node, 0, "result")
        check_edges_dimension_of_inputs(node, AllPositionals)
        check_edges_equivalence_of_inputs(node, AllPositionals)
        check_dtype_of_edges(node)
        check_edges_consistency_with_array(node, "result")
        copy_edges_from_inputs_to_outputs(node, 0, "result")
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
        copy_from_inputs_to_outputs(node, 0, "result")
        check_edges_dimension_of_inputs(node, AllPositionals)
        check_edges_equivalence_of_inputs(node, AllPositionals)
        check_dtype_of_edges(node)
        check_edges_consistency_with_array(node, "result")
        copy_edges_from_inputs_to_outputs(node, 0, "result")
    savegraph(graph, f"output/{testname}.png")
