from numpy import allclose, array
from pytest import mark

from dagflow.exception import TypeFunctionError
from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib import Array, Proxy


def test_Proxy(testname, debug_graph):
    np_array0 = array([1, 2, 3, 4, 5])
    np_array1 = array([0, 1, 2, 3, 4])

    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        array0 = Array("Array 0", np_array0)
        array1 = Array("Array 1", np_array1)
        proxy = Proxy("proxy node")
        array0 >> proxy
        array1.touch()

    array1 >> proxy

    assert allclose(proxy.get_data(), np_array0)
    savegraph(graph, f"output/{testname}-0.png", show="all")
    proxy.switch_input(1)
    assert allclose(proxy.get_data(), np_array1)
    savegraph(graph, f"output/{testname}-1.png", show="all")
