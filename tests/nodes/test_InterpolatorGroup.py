from matplotlib.pyplot import close
from matplotlib.pyplot import gca
from numpy import allclose
from numpy import finfo
from numpy import linspace
from numpy.random import seed
from numpy.random import shuffle

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib import Array
from dagflow.lib import InterpolatorGroup
from dagflow.lib import OneToOneNode
from dagflow.metanode import MetaNode
from dagflow.plot import plot_auto


class LinearF(OneToOneNode):
    """f(x) = k*x + b"""

    __slots__ = ("_k", "_b")

    def __init__(self, *args, k: float, b: float, **kwargs):
        super().__init__(*args, **kwargs)
        self._k = k
        self._b = b

    def _fcn(self):
        for inp, out in zip(self.inputs, self.outputs):
            out.data[:] = self._k * inp.data + self._b
        return list(self.outputs.iter_data())


def test_InterpolatorGroup(debug_graph, testname):
    k, b = 2.5, -3.5
    xlabel = "Nodes for the interpolator"
    seed(10)
    metaint = MetaNode()
    with Graph(debug=debug_graph, close=True) as graph:
        nc, nf = 10, 20
        coarseX = linspace(0, 10, nc + 1)
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
            labels={"interpolator": {"plottitle": "Interpolator", "axis": "y"}},
        )
        coarse >> metaint.inputs["coarse"]
        yc >> metaint.inputs[0]
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
