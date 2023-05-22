from random import choice
from string import ascii_lowercase

from dagflow.exception import InitializationError
from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib import Array
from dagflow.lib.InSegment import InSegment
from numpy import linspace
from numpy.random import shuffle
from pytest import mark, raises


@mark.parametrize("mode", ("left", "right"))
def test_insegment_01(debug_graph, testname, mode):
    with Graph(debug=debug_graph, close=True) as graph:
        nc, nf = 10, 100
        coarseX = linspace(0, 10, nc + 1)
        shuffle(coarseX)
        fineX = linspace(0, 10, nf + 1)
        shuffle(fineX)
        coarse = Array("coarse", coarseX)
        fine = Array("fine", fineX)
        insegment = InSegment("insegment", mode=mode)
        (coarse, fine) >> insegment
    res = coarseX.searchsorted(fineX, side=mode, sorter=coarseX.argsort())
    assert all(insegment.outputs[0].data == res)
    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("mode", ("left", "right"))
def test_insegment_02(debug_graph, testname, mode):
    with Graph(debug=debug_graph, close=True) as graph:
        nc, nf = 10, 100
        coarseX = linspace(0, 10, nc).reshape(2, nc // 2)
        shuffle(coarseX)
        fineX = linspace(0, 10, nf).reshape(2, nf // 2)
        shuffle(fineX)
        coarse = Array("coarse", coarseX)
        fine = Array("fine", fineX)
        insegment = InSegment("insegment", mode=mode)
        (coarse, fine) >> insegment
    res = coarseX.ravel().searchsorted(
        fineX.ravel(), side=mode, sorter=coarseX.ravel().argsort()
    )
    assert all(insegment.outputs[0].data.ravel() == res)
    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("mode", ("left", "right"))
def test_insegment_03(debug_graph, testname, mode):
    with Graph(debug=debug_graph, close=True) as graph:
        nc, nf = 10, 100
        coarseX = linspace(0, 10, nc).reshape(nc // 2, 2)
        shuffle(coarseX)
        fineX = linspace(0, 10, nf).reshape(nf // 2, 2)
        shuffle(fineX)
        coarse = Array("coarse", coarseX)
        fine = Array("fine", fineX)
        insegment = InSegment("insegment", mode=mode)
        (coarse, fine) >> insegment
    res = coarseX.ravel().searchsorted(
        fineX.ravel(), side=mode, sorter=coarseX.ravel().argsort()
    )
    assert all(insegment.outputs[0].data.ravel() == res)
    savegraph(graph, f"output/{testname}.png")


def test_insegment_exception(debug_graph):
    with Graph(debug=debug_graph, close=False):
        with raises(InitializationError):
            InSegment(
                "insegment",
                mode="".join(choice(ascii_lowercase) for _ in range(5)),
            )
