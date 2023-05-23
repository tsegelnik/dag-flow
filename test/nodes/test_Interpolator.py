from dagflow.exception import InitializationError
from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib import Array
from dagflow.lib.Interpolator import Interpolator
from dagflow.lib.SegmentIndex import SegmentIndex
from dagflow.lib.trigonometry import Sin
from numpy import allclose, exp, finfo, linspace, log, sin
from numpy.random import shuffle
from pytest import mark, raises


@mark.parametrize("k", (1.234, -0.578))
@mark.parametrize("b", (-5.432, 0.742))
def test_interpolation_linear_01(debug_graph, testname, k, b):
    with Graph(debug=debug_graph, close=True) as graph:
        nc, nf = 10, 20
        coarseX = linspace(0, 10, nc + 1)
        shuffle(coarseX)
        fineX = linspace(-2, 12, nf + 1)
        shuffle(fineX)
        ycX = k * coarseX + b
        coarse = Array("coarse", coarseX)
        fine = Array("fine", fineX)
        yc = Array("yc", ycX)
        segmentIndex = SegmentIndex("segmentIndex")
        interpolator = Interpolator("interpolator", method="linear")
        (coarse, fine) >> segmentIndex
        (coarse, yc, fine, segmentIndex.outputs[0]) >> interpolator
    assert allclose(
        interpolator.outputs[0].data,
        k * fineX + b,
        atol=finfo("d").resolution * 2,
        rtol=0,
    )
    savegraph(graph, f"output/{testname}.png")


def test_interpolation_linear_02(debug_graph, testname):
    with Graph(debug=debug_graph, close=True) as graph:
        nc, nf = 20, 20
        coarseX = linspace(-0.05, 0.05, nc + 1)
        shuffle(coarseX)
        fineX = linspace(-0.1, 0.1, nf + 1)
        shuffle(fineX)
        coarse = Array("coarse", coarseX)
        fine = Array("fine", fineX)
        ssin = Sin("sin")
        segmentIndex = SegmentIndex("segmentIndex")
        interpolator = Interpolator("interpolator", method="linear")
        (coarse, fine) >> segmentIndex
        coarse >> ssin
        (
            coarse,
            ssin.outputs[0],
            fine,
            segmentIndex.outputs[0],
        ) >> interpolator
    assert allclose(
        interpolator.outputs[0].data,
        sin(fineX),
        atol=1e-4,
    )
    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("shape", ((2, 10), (10, 2)))
def test_interpolation_ndim(debug_graph, testname, shape):
    with Graph(debug=debug_graph, close=True) as graph:
        nc, nf = 20, 20
        coarseX = linspace(-0.05, 0.05, nc).reshape(shape)
        shuffle(coarseX)
        fineX = linspace(-0.1, 0.1, nf).reshape(shape)
        shuffle(fineX)
        coarse = Array("coarse", coarseX)
        fine = Array("fine", fineX)
        ssin = Sin("sin")
        segmentIndex = SegmentIndex("segmentIndex")
        interpolator = Interpolator("interpolator", method="linear")
        (coarse, fine) >> segmentIndex
        coarse >> ssin
        (
            coarse,
            ssin.outputs[0],
            fine,
            segmentIndex.outputs[0],
        ) >> interpolator
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
        shuffle(coarseX)
        fineX = linspace(1e-2, 1e2, nf + 1)
        shuffle(fineX)
        ycX = log(k * coarseX + b)
        coarse = Array("coarse", coarseX)
        fine = Array("fine", fineX)
        yc = Array("yc", ycX)
        segmentIndex = SegmentIndex("segmentIndex")
        interpolator = Interpolator("interpolator", method="log")
        (coarse, fine) >> segmentIndex
        (coarse, yc, fine, segmentIndex.outputs[0]) >> interpolator
    assert allclose(
        interpolator.outputs[0].data,
        log(k * fineX + b),
        atol=finfo("d").resolution * 10,
        rtol=0,
    )
    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("k", (0.234, 1.578))
@mark.parametrize("b", (1.432, 0.742))
def test_interpolation_logx_01(debug_graph, testname, k, b):
    with Graph(debug=debug_graph, close=True) as graph:
        nc, nf = 100, 100
        coarseX = linspace(1e-1, 1e1, nc + 1)
        shuffle(coarseX)
        fineX = linspace(1e-2, 1e2, nf + 1)
        shuffle(fineX)
        ycX = k * log(coarseX) + b
        coarse = Array("coarse", coarseX)
        fine = Array("fine", fineX)
        yc = Array("yc", ycX)
        segmentIndex = SegmentIndex("segmentIndex")
        interpolator = Interpolator("interpolator", method="logx")
        (coarse, fine) >> segmentIndex
        (coarse, yc, fine, segmentIndex.outputs[0]) >> interpolator
    assert allclose(
        interpolator.outputs[0].data,
        k * log(fineX) + b,
        atol=finfo("d").resolution * 10,
        rtol=0,
    )
    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("k", (0.234, 1.578))
@mark.parametrize("b", (1.432, 0.742))
def test_interpolation_exp_01(debug_graph, testname, k, b):
    with Graph(debug=debug_graph, close=True) as graph:
        nc, nf = 100, 100
        coarseX = linspace(0, 0.5, nc + 1)
        ycX = exp(k * coarseX + b)
        fineX = linspace(0, 1, nf + 1)
        shuffle(fineX)
        coarse = Array("coarse", coarseX)
        fine = Array("fine", fineX)
        yc = Array("yc", ycX)
        segmentIndex = SegmentIndex("segmentIndex")
        interpolator = Interpolator("interpolator", method="exp")
        (coarse, fine) >> segmentIndex
        (coarse, yc, fine, segmentIndex.outputs[0]) >> interpolator
    assert allclose(
        interpolator.outputs[0].data,
        exp(k * fineX + b),
        atol=finfo("d").resolution * 100,
        rtol=0,
    )
    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("strategy", ("constant", "nearestedge"))
@mark.parametrize("fillvalue", (0, 1))
@mark.parametrize("k", (1.234, -0.578))
def test_interpolation_extrapolation_strategy(
    debug_graph, testname, k, strategy, fillvalue
):
    b = 5.137
    with Graph(debug=debug_graph, close=True) as graph:
        nc, nf = 10, 20
        coarseX = linspace(0, 10, nc + 1)
        shuffle(coarseX)
        fineX = linspace(-2, 12, nf + 1)
        shuffle(fineX)
        ycX = k * coarseX + b
        coarse = Array("coarse", coarseX)
        fine = Array("fine", fineX)
        yc = Array("yc", ycX)
        segmentIndex = SegmentIndex("segmentIndex")
        interpolator = Interpolator(
            "interpolator",
            method="linear",
            underflow=strategy,
            overflow=strategy,
            fillvalue=fillvalue,
        )
        (coarse, fine) >> segmentIndex
        (coarse, yc, fine, segmentIndex.outputs[0]) >> interpolator
    res = []
    for x in fineX:
        if x >= 0 and x <= 10:
            res.append(k * x + b)
        elif strategy == "nearestedge":
            if x < 0:
                res.append(b)
            else:
                res.append(k* 10 + b)
        elif strategy == "constant":
            res.append(fillvalue)
    assert allclose(
        interpolator.outputs[0].data,
        res,
        atol=finfo("d").resolution * 5,
        rtol=0,
    )
    savegraph(graph, f"output/{testname}.png")


def test_interpolation_exception(debug_graph):
    from random import choice
    from string import ascii_lowercase

    with Graph(debug=debug_graph, close=False):
        with raises(InitializationError):
            Interpolator(
                "interpolator",
                method="".join(choice(ascii_lowercase) for _ in range(5)),
            )

        with raises(InitializationError):
            Interpolator(
                "interpolator",
                underflow="".join(choice(ascii_lowercase) for _ in range(5)),
            )

        with raises(InitializationError):
            Interpolator(
                "interpolator",
                overflow="".join(choice(ascii_lowercase) for _ in range(5)),
            )
