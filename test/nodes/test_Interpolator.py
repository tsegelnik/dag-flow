from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib import Array
from dagflow.lib.InSegment import InSegment
from numpy import linspace, searchsorted
from numpy.random import shuffle
from pytest import mark


@mark.parametrize("mode", ("left", "right"))
def test_insegment(debug_graph, testname, mode):
    with Graph(debug=debug_graph, close=True) as graph:
        nc, nf = 10, 100
        coarseX = linspace(0, 10, nc + 1)
        fineX = linspace(0, 10, nf + 1)
        shuffle(fineX)
        coarse = Array("coarse", coarseX)
        fine = Array("fine", fineX)
        insegment = InSegment("insegment", mode=mode)
        (coarse, fine) >> insegment
    res = searchsorted(coarseX, fineX, mode)
    assert all(insegment.outputs[0].data == res)
    savegraph(graph, f"output/{testname}.png")
