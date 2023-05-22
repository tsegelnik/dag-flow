from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib import Array
from dagflow.lib.InSegment import InSegment
from dagflow.lib.Interpolator import Interpolator
from dagflow.lib.trigonometry import Sin
from numpy import allclose, finfo, linspace, log, searchsorted, sin
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


@mark.parametrize("k", (1.234, -0.578))
@mark.parametrize("b", (-5.432, 0.742))
def test_interpolation_linear_01(debug_graph, testname, k, b):
    with Graph(debug=debug_graph, close=True) as graph:
        nc, nf = 10, 20
        coarseX = linspace(0, 10, nc + 1)
        ycX = k * coarseX + b
        fineX = linspace(-2, 12, nf + 1)
        shuffle(fineX)
        coarse = Array("coarse", coarseX)
        fine = Array("fine", fineX)
        yc = Array("yc", ycX)
        insegment = InSegment("insegment")
        interpolator = Interpolator("interpolator", method="linear")
        (coarse, fine) >> insegment
        (coarse, yc, fine, insegment.outputs[0]) >> interpolator
    assert allclose(
        interpolator.outputs[0].data,
        k * fineX + b,
        rtol=finfo("d").resolution * 2,
        atol=0,
    )
    savegraph(graph, f"output/{testname}.png")


def test_interpolation_linear_02(debug_graph, testname):
    with Graph(debug=debug_graph, close=True) as graph:
        nc, nf = 20, 20
        coarseX = linspace(-0.05, 0.05, nc + 1)
        fineX = linspace(-0.1, 0.1, nf + 1)
        shuffle(fineX)
        coarse = Array("coarse", coarseX)
        fine = Array("fine", fineX)
        ssin = Sin("sin")
        insegment = InSegment("insegment")
        interpolator = Interpolator("interpolator", method="linear")
        (coarse, fine) >> insegment
        coarse >> ssin
        (coarse, ssin.outputs[0], fine, insegment.outputs[0]) >> interpolator
    assert allclose(
        interpolator.outputs[0].data,
        sin(fineX),
        atol=1e-4,
    )
    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("k", (10.234, 0.578))
@mark.parametrize("b", (15.432, 0.742))
def test_interpolation_log_01(debug_graph, testname, k, b):
    with Graph(debug=debug_graph, close=True) as graph:
        nc, nf = 100, 100
        coarseX = linspace(1e-1, 1e1, nc + 1)
        ycX = log(k * coarseX + b)
        fineX = linspace(1e-2, 1e2, nf + 1)
        shuffle(fineX)
        coarse = Array("coarse", coarseX)
        fine = Array("fine", fineX)
        yc = Array("yc", ycX)
        insegment = InSegment("insegment")
        interpolator = Interpolator("interpolator", method="log")
        (coarse, fine) >> insegment
        (coarse, yc, fine, insegment.outputs[0]) >> interpolator
    assert allclose(
        interpolator.outputs[0].data,
        log(k * fineX + b),
        rtol=finfo("d").resolution * 10,
        atol=0,
    )
    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("k", (0.234, 1.578))
@mark.parametrize("b", (1.432, 0.742))
def test_interpolation_logx_01(debug_graph, testname, k, b):
    with Graph(debug=debug_graph, close=True) as graph:
        nc, nf = 100, 100
        coarseX = linspace(1e-1, 1e1, nc + 1)
        ycX = k * log(coarseX) + b
        fineX = linspace(1e-2, 1e2, nf + 1)
        shuffle(fineX)
        coarse = Array("coarse", coarseX)
        fine = Array("fine", fineX)
        yc = Array("yc", ycX)
        insegment = InSegment("insegment")
        interpolator = Interpolator("interpolator", method="logx")
        (coarse, fine) >> insegment
        (coarse, yc, fine, insegment.outputs[0]) >> interpolator
    assert allclose(
        interpolator.outputs[0].data,
        k * log(fineX) + b,
        rtol=finfo("d").resolution * 10,
        atol=0,
    )
    savegraph(graph, f"output/{testname}.png")
