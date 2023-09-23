#!/usr/bin/env python

from numpy import allclose, finfo, geomspace, linspace, ndarray, ones_like, zeros_like
from matplotlib import pyplot as plt

from dagflow.graph import Graph
from dagflow.lib import Array
from dagflow.lib import BinCenter
from dagflow.graphviz import savegraph


def test_BinCenter_01(testname, debug_graph):
    array1 = linspace(0, 100, 26, dtype="f")
    array2 = geomspace(1e-2, 1e2, 10, dtype="d")

    with Graph(close=True, debug=debug_graph) as graph:
        arr1 = Array("linspace", array1)
        arr2 = Array("geomspace", array2)
        bc = BinCenter("centers")
        arr1 >> bc
        arr2 >> bc("geomspace")

    def bincenter(x: ndarray):
        return (x[1:] + x[:-1]) / 2.0

    def bincenter2(x: ndarray):
        return x[:-1] + (x[1:] - x[:-1]) / 2.0

    res1 = bc.outputs[0].data
    res2 = bc.outputs[1].data

    assert res1.dtype == "f"
    assert (res1 == ((array1 + 4 + array1) / 2)[:-1]).all()
    assert (res1 == bincenter(array1)).all()
    assert (res1 == bincenter2(array1)).all()

    assert res2.dtype == "d"
    assert (res2 == bincenter(array2)).all()
    assert allclose(res2, bincenter2(array2), atol=finfo("d").precision)

    plt.scatter(array1, zeros_like(array1), label="linspace edges")
    plt.scatter(res1, zeros_like(res1), marker="+", label="linspace centers")

    plt.scatter(array2, ones_like(array2), label="geomspace edges")
    plt.scatter(res2, ones_like(res2), marker="+", label="geomspace centers")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig(f"output/{testname}-plot.png")
    plt.close()

    savegraph(graph, f"output/{testname}.png")
