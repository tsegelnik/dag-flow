from matplotlib.pyplot import close, gca
from numpy import allclose, array, finfo, linspace
from numpy.random import seed, shuffle
from pytest import mark

from dagflow.core.graph import Graph
from dagflow.plot.graphviz import savegraph
from dagflow.lib.common import Array
from dagflow.lib.interpolation import Interpolator
from dagflow.lib.linalg import LinearFunction
from dagflow.core.meta_node import MetaNode
from dagflow.plot.plot import plot_auto


@mark.parametrize("dtype", ("d", "f"))
def test_Interpolator(debug_graph, testname, dtype):
    a, b = 2.5, -3.5
    xlabel = "Nodes for the interpolator"
    seed(10)
    metaint = MetaNode()
    with Graph(debug=debug_graph, close_on_exit=True) as graph:
        nc, nf = 10, 20
        coarseX = linspace(0, 10, nc + 1)
        fineX = linspace(-2, 12, nf + 1)
        shuffle(fineX)
        ycX = a * coarseX + b
        coarse = Array("coarse", coarseX, label=xlabel, dtype=dtype)
        fine = Array(
            "fine",
            fineX,
            label={"axis": xlabel},
            dtype=dtype,
        )
        yc = Array("yc", ycX, dtype=dtype)
        metaint = Interpolator(
            method="linear",
            labels={"interpolator": {"plottitle": "Interpolator", "axis": "y"}},
        )
        coarse >> metaint.inputs["coarse"]
        yc >> metaint.inputs[0]
        fine >> metaint.inputs["fine"]

        fcheck = LinearFunction("a*x+b")
        A = Array("a", array([a], dtype=dtype))
        B = Array("b", array([b], dtype=dtype))
        A >> fcheck("a")
        B >> fcheck("b")
        fine >> fcheck

        metaint.print()

    assert allclose(
        metaint.outputs[0].data,
        fcheck.outputs[0].data,
        rtol=0,
        atol=finfo(dtype).resolution * 2,
    )
    assert metaint.outputs[0].dd.axes_meshes == (fine["array"],)

    plot_auto(metaint)
    ax = gca()
    assert ax.get_xlabel() == xlabel
    assert ax.get_ylabel() == "y"
    assert ax.get_title() == "Interpolator"
    close()

    savegraph(graph, f"output/{testname}.pdf", show="all")
