from numpy import allclose, array

from pytest import raises
from dagflow.exception import ClosedGraphError
from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib.base import Array, Proxy


def test_Proxy_several_inputs(testname, debug_graph):
    np_array0 = array([1, 2, 3, 4, 5])
    np_array1 = array([0, 1, 2, 3, 4])

    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        array0 = Array("Array 0", np_array0)
        proxy = Proxy("proxy node")
        array0 >> proxy

    assert allclose(proxy.get_data(), np_array0)
    savegraph(graph, f"output/{testname}-0.png", show="all")

    with graph:
        graph.open(open_nodes=True)
        array1 = Array("Array 1", np_array1)
        array1 >> proxy
        proxy.open()
        proxy.close()

    proxy.switch_input(1)
    assert allclose(proxy.get_data(), np_array1)
    savegraph(graph, f"output/{testname}-1.png", show="all")

    proxy.switch_input(0)
    assert allclose(proxy.get_data(), np_array0)
    savegraph(graph, f"output/{testname}-2.png", show="all")


def test_Proxy_closed_graph_positional_input():
    np_array0 = array([1, 2, 3, 4, 5])
    np_array1 = array([0, 1, 2, 3, 4])

    with Graph(close_on_exit=True):
        array0 = Array("Array 0", np_array0)
        proxy = Proxy("proxy node")
        array0 >> proxy

    array1 = Array("Array 1", np_array1)
    with raises(ClosedGraphError) as e_info:
        array1 >> proxy


def test_Proxy_closed_graph_named_input():
    np_array0 = array([1, 2, 3, 4, 5])

    with Graph(close_on_exit=True):
        array0 = Array("Array 0", np_array0)
        proxy = Proxy("proxy node")
        array0 >> proxy

    with raises(ClosedGraphError) as e_info:
        new_input = proxy("new_input")
