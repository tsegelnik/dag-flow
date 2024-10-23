from pytest import raises

from dagflow.exception import TypeFunctionError
from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib.arithmetic import Sum
from dagflow.lib.common import Array
from dagflow.lib.common import Dummy
from dagflow.type_functions import check_has_inputs, check_inputs_number


def test_inputs_number_00(debug_graph):
    with Graph(close_on_exit=True, debug=debug_graph):
        node = Dummy("node")
    with raises(TypeFunctionError):
        check_has_inputs(node)
    check_inputs_number(node, 0)


def test_inputs_number_01(testname, debug_graph):
    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        arr1 = Array("arr1", [1, 2, 3])
        arr2 = Array("arr2", [3, 2, 1])
        ssum = Sum("sum")
        (arr1, arr2) >> ssum
    check_has_inputs(ssum, (0, 1))
    check_inputs_number(ssum, 2)
    with raises(TypeFunctionError):
        check_has_inputs("arr3")
    savegraph(graph, f"output/{testname}.png")
