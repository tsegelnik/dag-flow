from matplotlib import pyplot as plt
from numpy import allclose, finfo, geomspace, linspace, ndarray, ones_like, zeros_like
from pytest import mark

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib.axis import BinCenter
from dagflow.lib.base import Array


@mark.parametrize("dtype", ("d", "f"))
def test_BinCenter_01(testname, debug_graph, dtype):
    array1 = linspace(0, 100, 26, dtype=dtype)
    array2 = geomspace(1e-2, 1e2, 10, dtype=dtype)

    with Graph(close_on_exit=True, debug=debug_graph) as graph:
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

    assert res1.dtype == dtype
    assert (res1 == ((array1 + 4 + array1) / 2)[:-1]).all()
    assert (res1 == bincenter(array1)).all()
    assert (res1 == bincenter2(array1)).all()

    assert res2.dtype == dtype
    assert (res2 == bincenter(array2)).all()
    assert allclose(res2, bincenter2(array2), atol=finfo(dtype).resolution)

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
