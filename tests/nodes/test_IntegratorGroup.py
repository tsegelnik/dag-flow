from matplotlib.pyplot import close
from matplotlib.pyplot import gca
from numpy import allclose
from numpy import linspace
from numpy import pi

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib import Array
from dagflow.lib import Cos
from dagflow.lib import IntegratorGroup
from dagflow.lib import Sin
from dagflow.metanode import MetaNode
from dagflow.plot import plot_auto


def test_IntegratorGroup_trap(debug_graph):
    metaint = MetaNode()

    xlabel = "Edges for the integrator"
    with Graph(debug=debug_graph, close_on_exit=True) as graph:
        npoints = 10
        edges = Array("edges", linspace(0, pi, npoints + 1), label={"axis": xlabel})
        ordersX = Array("ordersX", [1000] * npoints, edges=edges["array"])
        A = Array("A", edges._data[:-1])
        B = Array("B", edges._data[1:])

        metaint = IntegratorGroup(
            "trap",
            labels={"integrator": {"plottitle": "Integrator", "axis": "integral"}},
        )

        cosf = Cos("cos")
        sinf = Sin("sin")
        ordersX >> metaint.inputs["ordersX"]

        metaint.outputs["x"] >> (cosf(), sinf())

        (cosf.outputs[0], sinf.outputs[0]) >> metaint

        sincheck = Sin("sin")
        coscheck = Cos("cos")
        A >> (sincheck(), coscheck())
        B >> (sincheck(), coscheck())

        metaint.print()
    res1 = sincheck.outputs[1].data - sincheck.outputs[0].data
    res2 = -coscheck.outputs[1].data + coscheck.outputs[0].data
    assert allclose(metaint.outputs[0].data, res1, rtol=0, atol=1e-2)
    assert allclose(metaint.outputs[1].data, res2, rtol=0, atol=1e-2)
    assert metaint.outputs[0].dd.axes_edges == [edges["array"]]
    assert metaint.outputs[1].dd.axes_edges == [edges["array"]]

    plot_auto(metaint)
    ax = gca()
    assert ax.get_xlabel() == xlabel
    assert ax.get_ylabel() == "integral"
    assert ax.get_title() == "Integrator"
    close()

    savegraph(graph, "output/test_Integrator_trap_meta.pdf", show="all")


# TODO: fix the problem with replication connection
# def test_Integrator_gl2d(debug_graph, testname):
#    vecF0 = vectorize(lambda x: 4 * x**3 + 3 * x**2 + 2 * x - 1)
#    vecFres = vectorize(lambda x: x**4 + x**3 + x**2 - x)
#
#    class PolynomialRes(OneToOneNode):
#        def _fcn(self):
#            for inp, out in zip(self.inputs, self.outputs):
#                out.data[:] = vecFres(inp.data)
#            return list(self.outputs.iter_data())
#
#    class Polynomial1(ManyToOneNode):
#        def _fcn(self):
#            self.outputs["result"].data[:] = vecF0(self.inputs[1].data) * vecF0(
#                self.inputs[0].data
#            )
#            return list(self.outputs.iter_data())
#
#    metaint = MetaNode()
#    with Graph(debug=debug_graph, close_on_exit=True) as graph:
#        npointsX, npointsY = 10, 20
#        edgesX = Array(
#            "edgesX",
#            linspace(0, 10, npointsX + 1),
#            label={"axis": "Label for axis X"},
#        )
#        edgesY = Array(
#            "edgesY",
#            linspace(2, 12, npointsY + 1),
#            label={"axis": "Label for axis Y"},
#        )
#        metaint, _ = IntegratorGroup.replicate(
#            "2d",
#            labels={"integrator": {"plottitle": "Integrator", "axis": "integral"}},
#            replicate_outputs=("A", "B", "C"),
#        )
#        assert len(metaint._nodes) == 4  # 1 sampler + 3 integrators
#
#        ordersX = Array("ordersX", [2] * npointsX, edges=edgesX["array"])
#        ordersY = Array("ordersY", [2] * npointsY, edges=edgesY["array"])
#        x0, y0 = meshgrid(edgesX._data[:-1], edgesY._data[:-1], indexing="ij")
#        x1, y1 = meshgrid(edgesX._data[1:], edgesY._data[1:], indexing="ij")
#        X0, X1 = Array("X0", x0), Array("X1", x1)
#        Y0, Y1 = Array("Y0", y0), Array("Y1", y1)
#
#        poly0 = Polynomial1("poly0")
#        polyres = PolynomialRes("polyres")
#        ordersX >> metaint("ordersX")
#        metaint.outputs["x"] >> poly0
#        metaint.outputs["y"] >> poly0
#        X0 >> polyres
#        X1 >> polyres
#        Y0 >> polyres
#        Y1 >> polyres
#        poly0.outputs[0] >> metaint
#        ordersY >> metaint("ordersY")
#        metaint.print()
#        tuple(node.print() for node in metaint._nodes)
#    res = (polyres.outputs[1].data - polyres.outputs[0].data) * (
#        polyres.outputs[3].data - polyres.outputs[2].data
#    )
#    assert allclose(metaint.outputs[0].data, res, atol=1e-10)
#    assert metaint.outputs[0].dd.axes_edges == [
#        edgesX["array"],
#        edgesY["array"],
#    ]
#
#    savegraph(graph, f"output/{testname}.png", show="all")
