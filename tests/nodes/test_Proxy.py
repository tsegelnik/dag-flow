from numpy import allclose, array
from pytest import mark

from dagflow.exception import TypeFunctionError
from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib import Array, Proxy

from dagflow.logger import DEBUG
from dagflow.logger import set_level

set_level(DEBUG)


def test_Proxy(testname, debug_graph):
    np_array0 = array([1, 2, 3, 4, 5])
    np_array1 = array([0, 1, 2, 3, 4])

    with Graph(close_on_exit=True) as graph:
        array0 = Array("Array 0", np_array0)
        proxy = Proxy("proxy node")
        array0 >> proxy
        # array1 = Array("Array 1", np_array1)
        # array1 >> proxy

    assert allclose(proxy.get_data(), np_array0)
    savegraph(graph, f"output/{testname}-0.png", show="all")

    with graph:
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

