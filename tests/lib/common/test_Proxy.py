from numpy import allclose, arange, array
from pytest import raises

from dagflow.core.exception import ClosedGraphError
from dagflow.core.graph import Graph
from dagflow.lib.common import Array, Proxy
from dagflow.plot.graphviz import savegraph


def test_Proxy_several_inputs(testname, debug_graph):
    arrays = [arange(5) + i for i in range(8)]

    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        array0 = Array("Array 0", arrays[0])
        proxy = Proxy("proxy node")
        array0 >> proxy

    assert allclose(proxy.get_data(), arrays[0], atol=0, rtol=0)
    savegraph(graph, f"output/{testname}-0.png", show="all")

    with graph:
        graph.open(open_nodes=True)
        for i, array in enumerate(arrays[1:], 1):
            array1 = Array(f"Array {i}", array)
            array1 >> proxy

    for i, array in enumerate(arrays):
        proxy.switch_input(i)
        assert proxy.tainted
        assert allclose(proxy.get_data(), arrays[i], atol=0, rtol=0)

        savegraph(graph, f"output/{testname}-i.png", show="all")

    proxy.switch_input(0)
    assert proxy.tainted
    assert allclose(proxy.get_data(), arrays[0], atol=0, rtol=0)
    savegraph(graph, f"output/{testname}-0-again.png", show="all")

    proxy.switch_input(0)
    assert not proxy.tainted
    assert allclose(proxy.get_data(), arrays[0], atol=0, rtol=0)
    savegraph(graph, f"output/{testname}-0-again.png", show="all")


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
