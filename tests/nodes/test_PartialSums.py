from numpy import allclose
from numpy import arange
from numpy import finfo
from numpy import linspace
from pytest import mark
from pytest import raises

from dagflow.exception import TypeFunctionError
from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib import Array
from dagflow.lib import PartialSums


@mark.parametrize("a", (arange(12, dtype="d") * i for i in (1, 2, 3)))
def test_PartialSums_01(testname, debug_graph, a):
    arrays_range = [0, 12], [0, 3], [4, 10], [11, 12]
    arrays_res = tuple(a[ranges[0] : ranges[1]].sum() for ranges in arrays_range)

    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        ranges = tuple(Array(f"range_{i}", arr) for i, arr in enumerate(arrays_range))
        arra = Array("a", a)
        ps = PartialSums("partialsums")
        arra >> ps("array")
        ranges >> ps

    atol = finfo("d").resolution * 2
    assert ps.tainted is True
    assert all(
        allclose(output.data[0], res, rtol=0, atol=atol)
        for output, res in zip(ps.outputs, arrays_res)
    )
    assert ps.tainted is False

    savegraph(graph, f"output/{testname}.png", show="all")


@mark.parametrize("a", (arange(12, dtype="d") * i for i in (1, 2, 3)))
def test_PartialSums_edges(debug_graph, a):
    arrays_range = [0, 12], [0, 3], [4, 10], [11, 12]
    with Graph(close_on_exit=False, debug=debug_graph) as graph:
        edges = Array("edges", linspace(0, 13, 13))
        arra = Array("a", a, edges=edges["array"])
        ranges = tuple(Array(f"range_{i}", arr) for i, arr in enumerate(arrays_range))
        ps = PartialSums("partialsums")
        arra >> ps("array")
        ranges >> ps
    with raises(TypeFunctionError):
        graph.close()
