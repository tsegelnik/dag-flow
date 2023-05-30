#!/usr/bin/env python

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib.Array import Array
from dagflow.lib.InverseSquareLaw import InverseSquareLaw
from numpy import allclose, arange, finfo, pi
from pytest import mark


@mark.parametrize("dtype", ("d", "f"))
def test_InverseSquareLaw_01(debug_graph, testname, dtype):
    arrays_in = tuple(arange(12, dtype=dtype) * i for i in (1, 2, 3))
    res_all = tuple(0.5 / pi / a**2 for a in arrays_in)

    with Graph(close=True, debug=debug_graph) as graph:
        arrays = tuple(Array("test", array_in) for array_in in arrays_in)
        isl = InverseSquareLaw("InvSqLaw")
        arrays >> isl

    atol = finfo(dtype).precision * 2
    assert isl.tainted is True
    assert all(output.dd.dtype == dtype for output in isl.outputs)
    assert all(
        allclose(output.data, res, rtol=0, atol=atol)
        for output, res in zip(isl.outputs, res_all)
    )
    assert isl.tainted is False

    savegraph(graph, f"output/{testname}.png")
