#!/usr/bin/env python

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib.Array import Array
from dagflow.lib.PartialSums import PartialSums
from numpy import allclose, arange, finfo
from pytest import mark


@mark.parametrize("a", (arange(12, dtype="d") * i for i in (1, 2, 3)))
def test_PartialSums_01(testname, debug_graph, a):
    arrays_range = [0, 12], [0, 3], [4, 10], [11, 12]
    arrays_res = tuple(a[ranges[0] : ranges[1]].sum() for ranges in arrays_range)

    with Graph(close=True, debug=debug_graph) as graph:
        arra = Array("a", a)
        ranges = tuple(Array(f"range_{i}", arr) for i, arr in enumerate(arrays_range))
        ps = PartialSums("partialsums")
        arra >> ps("a")
        ranges >> ps

    atol = finfo("d").precision * 2
    assert ps.tainted is True
    assert all(
        allclose(output.data[0], res, rtol=0, atol=atol)
        for output, res in zip(ps.outputs, arrays_res)
    )
    assert ps.tainted is False

    savegraph(graph, f"output/{testname}.png", show="all")
