#!/usr/bin/env python
from numpy import allclose
from numpy import arange
from numpy import finfo
from pytest import mark

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib import Array
from dagflow.lib import ArraySum


@mark.parametrize("a", (arange(12, dtype="d") * i for i in (1, 2, 3)))
def test_ArraySum_01(testname, debug_graph, a):
    a2 = a*(-1.1)
    array_res = a.sum()
    array2_res = a2.sum()

    with Graph(close=True, debug=debug_graph) as graph:
        arra = Array("a", a)
        arra2 = Array("a2", a2)
        arraysum = ArraySum("arraysum")
        arra >> arraysum
        arra2 >> arraysum

    atol = finfo("d").resolution * 2
    assert arraysum.tainted
    assert allclose(arraysum.get_data(0)[0], array_res, rtol=0, atol=atol)
    assert allclose(arraysum.get_data(1)[0], array2_res, rtol=0, atol=atol)
    assert arraysum.tainted is False

    savegraph(graph, f"output/{testname}.png", show="all")
