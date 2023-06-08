from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib.Array import Array
from dagflow.lib.InterpolatorGroup import InterpolatorGroup
from dagflow.lib.OneToOneNode import OneToOneNode
from dagflow.meta_node import MetaNode
from dagflow.plot import plot_auto
from matplotlib.pyplot import close, gca
from numpy import allclose, finfo, linspace
from numpy.random import seed, shuffle


class LinearF(OneToOneNode):
    """f(x) = k*x + b"""

    __slots__ = ("_k", "_b")

    def __init__(self, *args, k: float, b: float, **kwargs):
        super().__init__(*args, **kwargs)
        self._k = k
        self._b = b

    def _fcn(self, _, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            out.data[:] = self._k * inp.data + self._b
        return list(outputs.iter_data())


def test_InterpolatorGroup(debug_graph, testname):
    k, b = 2.5, -3.5
    xlabel = "Nodes for the interpolator"
    seed(10)
    metaint = MetaNode()
    with Graph(debug=debug_graph, close=True) as graph:
        nc, nf = 10, 20
        coarseX = linspace(0, 10, nc + 1)
        shuffle(coarseX)
        fineX = linspace(-2, 12, nf + 1)
        shuffle(fineX)
        ycX = k * coarseX + b
        coarse = Array("coarse", coarseX, label=xlabel)
        fine = Array(
            "fine",
            fineX,
            label={"axis": xlabel},
        )
        yc = Array("yc", ycX)
        metaint = InterpolatorGroup(
            method="linear",
            labels={
                "interpolator": {"plottitle": "Interpolator", "axis": "y"}
            },
        )
        coarse >> metaint.inputs["coarse"]
        yc >> metaint.inputs["y"]
        fine >> metaint.inputs["fine"]

        fcheck = LinearF("k*x+b", k=k, b=b)
        fine >> fcheck

        metaint.print()

    assert allclose(
        metaint.outputs[0].data,
        fcheck.outputs[0].data,
        rtol=0,
        atol=finfo("d").resolution * 5,
    )
    assert metaint.outputs[0].dd.axes_meshes == (fine["array"],)

    plot_auto(metaint)
    ax = gca()
    assert ax.get_xlabel() == xlabel
    assert ax.get_ylabel() == "y"
    assert ax.get_title() == "Interpolator"
    close()

    savegraph(graph, f"output/{testname}.pdf", show="all")
