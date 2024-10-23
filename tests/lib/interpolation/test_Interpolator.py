from numpy import allclose, concatenate, exp, finfo, linspace, log, sin, where, zeros_like
from numpy.random import seed, shuffle
from pytest import mark, raises

from dagflow.exception import CalculationError, InitializationError
from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib.common import Array
from dagflow.lib.interpolation import Interpolator, SegmentIndex
from dagflow.lib.trigonometry import Sin


@mark.parametrize("k", (1.234, -0.578))
@mark.parametrize("b", (-5.432, 0.742))
@mark.parametrize("fine_x_mode", ("other", "same"))
def test_interpolation_linear_01(debug_graph, testname, k, b, fine_x_mode):
    seed(10)

    nc, nf = 10, 25
    coarseX = linspace(0, 10, nc + 1)

    match fine_x_mode:
        case "other":
            fineX = linspace(-2, 12, nf + 1)
        case "same":
            fineX = concatenate((coarseX, coarseX))
        case _:
            raise RuntimeError(fine_x_mode)
    shuffle(fineX)

    ycX = k * coarseX + b

    with Graph(debug=debug_graph, close_on_exit=True) as graph:
        coarse = Array("coarse", coarseX)
        fine = Array("fine", fineX)
        yc = Array("yc", ycX)
        segmentIndex = SegmentIndex("indexer")
        interpolator = Interpolator("interpolator", method="linear")

        (coarse, fine) >> segmentIndex
        yc >> interpolator
        coarse >> interpolator("coarse")
        fine >> interpolator("fine")
        segmentIndex.outputs[0] >> interpolator("indices")

    assert allclose(
        interpolator.outputs[0].data,
        k * fineX + b,
        atol=finfo("d").resolution * 10,
        rtol=0,
    )
    savegraph(graph, f"output/{testname}.png")


def test_interpolation_linear_02(debug_graph, testname):
    seed(10)

    nc, nf = 20, 45
    coarseX = linspace(-0.05, 0.05, nc + 1)
    fineX = linspace(-0.1, 0.1, nf + 1)
    shuffle(fineX)

    with Graph(debug=debug_graph, close_on_exit=True) as graph:
        coarse = Array("coarse", coarseX)
        fine = Array("fine", fineX)
        ssin = Sin("sin")
        segmentIndex = SegmentIndex("indexer")
        interpolator = Interpolator("interpolator", method="linear")

        (coarse, fine) >> segmentIndex
        coarse >> ssin
        ssin.outputs[0] >> interpolator
        coarse >> interpolator("coarse")
        fine >> interpolator("fine")
        segmentIndex.outputs[0] >> interpolator("indices")

    assert allclose(
        interpolator.outputs[0].data,
        sin(fineX),
        atol=1e-4,
    )
    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("shape", ((3, 15), (15, 3)))
def test_interpolation_ndim(debug_graph, testname, shape):
    seed(10)

    nc, nf = 20, 45
    coarseX = linspace(-0.05, 0.05, nc)  # .reshape(shape) # TODO: 2d coarse X/Y should be forbidden
    fineX = linspace(-0.1, 0.1, nf).reshape(shape)
    shuffle(fineX)

    with Graph(debug=debug_graph, close_on_exit=True) as graph:
        coarse = Array("coarse", coarseX)
        fine = Array("fine", fineX)
        ssin = Sin("sin")
        segmentIndex = SegmentIndex("indexer")
        interpolator = Interpolator("interpolator", method="linear")

        (coarse, fine) >> segmentIndex
        coarse >> ssin
        ssin.outputs[0] >> interpolator
        coarse >> interpolator("coarse")
        fine >> interpolator("fine")
        segmentIndex.outputs[0] >> interpolator("indices")

    assert allclose(
        interpolator.outputs[0].data,
        sin(fineX),
        atol=1e-4,
    )
    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("k", (10.234, 0.578))
@mark.parametrize("b", (15.432, 0.742))
@mark.parametrize("fine_x_mode", ("other", "same"))
def test_interpolation_log_01(debug_graph, testname, k, b, fine_x_mode):
    seed(10)

    nc, nf = 100, 251
    coarseX = linspace(1e-1, 1e1, nc + 1)
    match fine_x_mode:
        case "other":
            fineX = linspace(1e-2, 1e2, nf + 1)
        case "same":
            fineX = concatenate((coarseX, coarseX))
        case _:
            raise RuntimeError(fine_x_mode)

    shuffle(fineX)

    ycX = log(k * coarseX + b)

    with Graph(debug=debug_graph, close_on_exit=True) as graph:
        coarse = Array("coarse", coarseX)
        fine = Array("fine", fineX)
        yc = Array("yc", ycX)
        segmentIndex = SegmentIndex("indexer")
        interpolator = Interpolator("interpolator", method="log")

        (coarse, fine) >> segmentIndex
        yc >> interpolator
        coarse >> interpolator("coarse")
        fine >> interpolator("fine")
        segmentIndex.outputs[0] >> interpolator("indices")

    assert allclose(
        interpolator.outputs[0].data,
        log(k * fineX + b),
        atol=finfo("d").resolution * 100,
        rtol=0,
    )
    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("k", (0.234, 1.578))
@mark.parametrize("b", (1.432, 0.742))
@mark.parametrize("fine_x_mode", ("other", "same"))
def test_interpolation_logx_01(debug_graph, testname, k, b, fine_x_mode):
    seed(10)

    nc, nf = 100, 251
    coarseX = linspace(1e-1, 1e1, nc + 1)

    match fine_x_mode:
        case "other":
            fineX = linspace(1e-2, 1e2, nf + 1)
        case "same":
            fineX = concatenate((coarseX, coarseX))
        case _:
            raise RuntimeError(fine_x_mode)

    shuffle(fineX)
    ycX = k * log(coarseX) + b

    with Graph(debug=debug_graph, close_on_exit=True) as graph:
        coarse = Array("coarse", coarseX)
        fine = Array("fine", fineX)
        yc = Array("yc", ycX)
        segmentIndex = SegmentIndex("indexer")
        interpolator = Interpolator("interpolator", method="logx")

        (coarse, fine) >> segmentIndex
        yc >> interpolator
        coarse >> interpolator("coarse")
        fine >> interpolator("fine")
        segmentIndex.outputs[0] >> interpolator("indices")

    assert allclose(
        interpolator.outputs[0].data,
        k * log(fineX) + b,
        atol=finfo("d").resolution * 100,
        rtol=0,
    )
    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("k", (0.234, 1.578))
@mark.parametrize("b", (1.432, 0.742))
@mark.parametrize("fine_x_mode", ("other", "same"))
def test_interpolation_exp_01(debug_graph, testname, k, b, fine_x_mode):
    seed(10)

    nc, nf = 100, 251
    coarseX = linspace(0, 0.5, nc + 1)
    match fine_x_mode:
        case "other":
            fineX = linspace(0, 1, nf + 1)
        case "same":
            fineX = concatenate((coarseX, coarseX))
        case _:
            raise RuntimeError(fine_x_mode)

    shuffle(fineX)
    ycX = exp(k * coarseX + b)

    with Graph(debug=debug_graph, close_on_exit=True) as graph:
        coarse = Array("coarse", coarseX)
        fine = Array("fine", fineX)
        yc = Array("yc", ycX)
        segmentIndex = SegmentIndex("indexer")
        interpolator = Interpolator("interpolator", method="exp")

        (coarse, fine) >> segmentIndex
        yc >> interpolator
        coarse >> interpolator("coarse")
        fine >> interpolator("fine")
        segmentIndex.outputs[0] >> interpolator("indices")

    assert allclose(
        interpolator.outputs[0].data,
        exp(k * fineX + b),
        atol=finfo("d").resolution * 1e3,
        rtol=0,
    )
    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("k", (0.234, 1.578))
@mark.parametrize("b", (1.432, 0.742))
@mark.parametrize("method", ("left", "right", "nearest"))
@mark.parametrize("fine_x_mode", ("other", "same"))
@mark.parametrize("truncate_y", (False, True))
def test_interpolation_lrn_01(
    debug_graph, testname, k, b, method: str, fine_x_mode: str, truncate_y: bool
):
    seed(10)

    nc, nf = 100, 251
    coarseX = linspace(0, 0.5, nc + 1)
    ycX = exp(k * coarseX + b)

    if truncate_y:
        if method == "left":
            ycX = ycX[:-1]
        else:
            return

    match fine_x_mode:
        case "other":
            fineX = linspace(-0.1, 1, nf + 1)
        case "same":
            fineX = concatenate((coarseX, coarseX))
        case _:
            raise RuntimeError(fine_x_mode)

    shuffle(fineX)

    with Graph(debug=debug_graph, close_on_exit=True) as graph:
        coarse = Array("coarse", coarseX)
        fine = Array("fine", fineX)
        yc = Array("yc", ycX)
        segmentIndex = SegmentIndex("indexer")
        interpolator = Interpolator("interpolator", method=method)

        (coarse, fine) >> segmentIndex
        yc >> interpolator
        coarse >> interpolator("coarse")
        fine >> interpolator("fine")
        segmentIndex.outputs[0] >> interpolator("indices")

    compareA = interpolator.outputs[0].data

    compareB = zeros_like(fineX)
    idx = coarseX.searchsorted(fineX, side="right")
    are_below = idx < 1
    are_above = idx > nc
    are_in = (~are_below) * (~are_above)
    match method:
        case "left":
            compareB[are_in] = ycX[idx[are_in] - 1]
            compareB[are_below] = ycX[0]
            if truncate_y:
                compareB[are_above] = ycX[-1]
            else:
                compareB[are_above] = ycX[-2]
        case "right":
            compareB[are_in] = ycX[idx[are_in]]
            compareB[are_below] = ycX[1]
            compareB[are_above] = ycX[-1]
        case "nearest":
            xleft = coarseX[idx[are_in] - 1]
            xright = coarseX[idx[are_in]]
            yleft = ycX[idx[are_in] - 1]
            yright = ycX[idx[are_in]]
            xmid = fineX[are_in]
            compareB[are_in] = where((xmid - xleft) <= (xright - xmid), yleft, yright)
            compareB[are_below] = ycX[0]
            compareB[are_above] = ycX[-1]
        case _:
            raise RuntimeError(method)

    assert allclose(
        compareA,
        compareB,
        atol=0,
        rtol=0,
    )
    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("strategy", ("constant", "nearestedge"))
@mark.parametrize("fillvalue", (0, 1))
@mark.parametrize("k", (1.234, -0.578))
def test_interpolation_extrapolation_strategy(debug_graph, testname, k, strategy, fillvalue):
    seed(10)

    b = 5.137
    nc, nf = 10, 25
    coarseX = linspace(0, 10, nc + 1)
    fineX = linspace(-2, 12, nf + 1)
    shuffle(fineX)
    ycX = k * coarseX + b

    with Graph(debug=debug_graph, close_on_exit=True) as graph:
        coarse = Array("coarse", coarseX)
        fine = Array("fine", fineX)
        yc = Array("yc", ycX)
        segmentIndex = SegmentIndex("indexer")
        interpolator = Interpolator(
            "interpolator",
            method="linear",
            underflow=strategy,
            overflow=strategy,
            fillvalue=fillvalue,
        )

        (coarse, fine) >> segmentIndex
        yc >> interpolator
        coarse >> interpolator("coarse")
        fine >> interpolator("fine")
        segmentIndex.outputs[0] >> interpolator("indices")

    res = []
    for x in fineX:
        if x >= 0 and x <= 10:
            res.append(k * x + b)
        elif strategy == "nearestedge":
            if x < 0:
                res.append(b)
            else:
                res.append(k * 10 + b)
        elif strategy == "constant":
            res.append(fillvalue)

    assert allclose(
        interpolator.outputs[0].data,
        res,
        atol=finfo("d").resolution * 5,
        rtol=0,
    )
    savegraph(graph, f"output/{testname}.png")


def test_interpolation_exception_01(debug_graph):
    from random import choice
    from string import ascii_lowercase

    with Graph(debug=debug_graph, close_on_exit=False):
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


def test_interpolation_exception_02(debug_graph):
    b = 5.137
    k = 1.1
    nc, nf = 10, 25
    coarseX = linspace(0, 10, nc + 1)
    shuffle(coarseX)
    fineX = linspace(-2, 12, nf + 1)
    ycX = k * coarseX + b

    with Graph(debug=debug_graph, close_on_exit=True):
        coarse = Array("coarse", coarseX)
        fine = Array("fine", fineX)
        yc = Array("yc", ycX)
        segmentIndex = SegmentIndex("indexer")
        interpolator = Interpolator("interpolator", method="linear")

        (coarse, fine) >> segmentIndex
        yc >> interpolator
        coarse >> interpolator("coarse")
        fine >> interpolator("fine")
        segmentIndex.outputs[0] >> interpolator("indices")

    with raises(CalculationError):
        interpolator.outputs[0].data
