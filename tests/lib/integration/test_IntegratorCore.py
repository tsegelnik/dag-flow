from matplotlib.pyplot import close, subplots
from numpy import allclose, linspace, meshgrid, pi, vectorize
from pytest import mark, raises

from dagflow.core.exception import TypeFunctionError
from dagflow.core.graph import Graph
from dagflow.plot.graphviz import savegraph
from dagflow.lib.abstract import ManyToOneNode, OneToOneNode
from dagflow.lib.common import Array
from dagflow.lib.integration import IntegratorCore, IntegratorSampler
from dagflow.lib.trigonometry import Cos, Sin
from dagflow.plot.plot import plot_auto


@mark.parametrize("align", ("left", "center", "right"))
def test_IntegratorCore_rect_center(align, debug_graph, testname):
    with Graph(debug=debug_graph, close_on_exit=True) as graph:
        npoints = 10
        edges = Array("edges", linspace(0, pi, npoints + 1))
        ordersX = Array("ordersX", [1000] * npoints, edges=edges["array"])
        A = Array("A", edges._data[:-1])
        B = Array("B", edges._data[1:])
        sampler = IntegratorSampler("sampler", mode="rect", align=align)
        integrator = IntegratorCore("integrator")
        cosf = Cos("cos")
        sinf = Sin("sin")
        ordersX >> sampler("ordersX")
        sampler.outputs["x"] >> cosf
        A >> sinf
        B >> sinf
        sampler.outputs["weights"] >> integrator("weights")
        cosf.outputs[0] >> integrator
        ordersX >> integrator("ordersX")
    res = sinf.outputs[1].data - sinf.outputs[0].data
    assert allclose(integrator.outputs[0].data, res, atol=1e-4)
    integrator.taint()
    integrator.touch()
    assert allclose(integrator.outputs[0].data, res, atol=1e-4)
    assert integrator.outputs[0].dd.axes_edges == [edges["array"]]
    savegraph(graph, f"output/{testname}.png")


def test_IntegratorCore_trap(debug_graph, testname):
    with Graph(debug=debug_graph, close_on_exit=True) as graph:
        npoints = 10
        edges = Array("edges", linspace(0, pi, npoints + 1))
        ordersX = Array("ordersX", [1000] * npoints, edges=edges["array"])
        A = Array("A", edges._data[:-1])
        B = Array("B", edges._data[1:])
        sampler = IntegratorSampler("sampler", mode="trap")
        integrator = IntegratorCore("integrator")
        cosf = Cos("cos")
        sinf = Sin("sin")
        ordersX >> sampler("ordersX")
        sampler.outputs["x"] >> cosf
        A >> sinf
        B >> sinf
        sampler.outputs["weights"] >> integrator("weights")
        cosf.outputs[0] >> integrator
        ordersX >> integrator("ordersX")
    res = sinf.outputs[1].data - sinf.outputs[0].data
    assert allclose(integrator.outputs[0].data, res, atol=1e-2)
    assert integrator.outputs[0].dd.axes_edges == [edges["array"]]
    savegraph(graph, f"output/{testname}.png")


def f0(x: float) -> float:
    return 4 * x**3 + 3 * x**2 + 2 * x - 1


def fres(x: float) -> float:
    return x**4 + x**3 + x**2 - x


vecF0 = vectorize(f0)
vecFres = vectorize(fres)


class Polynomial0(OneToOneNode):
    def _fcn(self):
        for inp, out in zip(self.inputs, self.outputs):
            out.data[:] = vecF0(inp.data)


class PolynomialRes(OneToOneNode):
    def _fcn(self):
        for inp, out in zip(self.inputs, self.outputs):
            out.data[:] = vecFres(inp.data)


def test_IntegratorCore_gl1d(debug_graph, testname):
    with Graph(debug=debug_graph, close_on_exit=True) as graph:
        npoints = 10
        edges = Array("edges", linspace(0, 10, npoints + 1))
        ordersX = Array("ordersX", [2] * npoints, edges=edges["array"])
        A = Array("A", edges._data[:-1])
        B = Array("B", edges._data[1:])
        sampler = IntegratorSampler("sampler", mode="gl")
        integrator = IntegratorCore("integrator")
        poly0 = Polynomial0("poly0")
        polyres = PolynomialRes("polyres")
        ordersX >> sampler("ordersX")
        sampler.outputs["x"] >> poly0
        A >> polyres
        B >> polyres
        sampler.outputs["weights"] >> integrator("weights")
        poly0.outputs[0] >> integrator
        ordersX >> integrator("ordersX")
    res = polyres.outputs[1].data - polyres.outputs[0].data
    assert allclose(integrator.outputs[0].data, res, atol=1e-10)
    assert integrator.outputs[0].dd.axes_edges == [edges["array"]]
    savegraph(graph, f"output/{testname}.png")


def test_IntegratorCore_gl2d(debug_graph, testname):
    class Polynomial1(ManyToOneNode):
        scale = 1.0

        def _fcn(self):
            self.outputs["result"].data[:] = (
                self.scale * vecF0(self.inputs[1].data) * vecF0(self.inputs[0].data)
            )

    with Graph(debug=debug_graph, close_on_exit=True) as graph:
        npointsX, npointsY = 10, 20
        edgesX = Array(
            "edgesX",
            linspace(0, 10, npointsX + 1),
            label={"axis": "Label for axis X"},
        )
        edgesY = Array(
            "edgesY",
            linspace(2, 12, npointsY + 1),
            label={"axis": "Label for axis Y"},
        )
        ordersX = Array("ordersX", [2] * npointsX, edges=edgesX["array"])
        ordersY = Array("ordersY", [2] * npointsY, edges=edgesY["array"])
        x0, y0 = meshgrid(edgesX._data[:-1], edgesY._data[:-1], indexing="ij")
        x1, y1 = meshgrid(edgesX._data[1:], edgesY._data[1:], indexing="ij")
        X0, X1 = Array("X0", x0), Array("X1", x1)
        Y0, Y1 = Array("Y0", y0), Array("Y1", y1)
        sampler = IntegratorSampler("sampler", mode="2d")
        integrator = IntegratorCore(
            "integrator",
            label={
                "plottitle": f"IntegratorCore test: {testname}",
                "axis": "integral",
            },
        )
        poly0 = Polynomial1("poly0")
        polyres = PolynomialRes("polyres")
        ordersX >> sampler("ordersX")
        ordersY >> sampler("ordersY")
        sampler.outputs["x"] >> poly0
        sampler.outputs["y"] >> poly0
        X0 >> polyres
        X1 >> polyres
        Y0 >> polyres
        Y1 >> polyres
        sampler.outputs["weights"] >> integrator("weights")
        poly0.outputs[0] >> integrator
        ordersX >> integrator("ordersX")
        ordersY >> integrator("ordersY")

    res = (polyres.outputs[1].data - polyres.outputs[0].data) * (
        polyres.outputs[3].data - polyres.outputs[2].data
    )
    assert allclose(integrator.outputs[0].data, res, atol=1e-10)
    integrator.taint()
    integrator.touch()
    assert allclose(integrator.outputs[0].data, res, atol=1e-10)
    assert integrator.outputs[0].dd.axes_edges == [
        edgesX["array"],
        edgesY["array"],
    ]

    subplots(1, 1, subplot_kw={"projection": "3d"})
    plot_auto(integrator, method="bar3d", colorbar=True)

    subplots(1, 1)
    plot_auto(integrator, method="pcolormesh", colorbar=True)

    subplots(1, 1)
    plot_auto(integrator, method="pcolor", colorbar=True)

    subplots(1, 1)
    plot_auto(integrator, method="pcolorfast", colorbar=True)

    subplots(1, 1)
    plot_auto(integrator, method="imshow", colorbar=True)

    subplots(1, 1)
    plot_auto(integrator, method="matshow", colorbar=True)

    subplots(1, 1)
    plot_auto(integrator, method="matshow", extent=None, colorbar=True)

    for _ in range(7):
        close()

    poly0.scale = 2.2
    poly0.taint()
    assert allclose(integrator.outputs[0].data, res * poly0.scale, atol=1e-10)

    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("dropdim", (True, False))
def test_IntegratorCore_gl2to1d_x(debug_graph, testname, dropdim):
    class Polynomial21(ManyToOneNode):
        def _fcn(self):
            self.outputs["result"].data[:] = vecF0(self.inputs[1].data) * vecF0(self.inputs[0].data)

    with Graph(debug=debug_graph, close_on_exit=True) as graph:
        npointsX = 20
        edgesX = Array(
            "edgesX",
            linspace(2, 12, npointsX + 1),
            label={"axis": "Label for axis X"},
        )
        edgesY = Array(
            "edgesY",
            [0, 1],
            label={"axis": "Label for axis Y"},
        )
        ordersX = Array("ordersX", [2] * npointsX, edges=edgesX["array"])
        ordersY = Array("ordersY", [2], edges=edgesY["array"])
        x0, y0 = meshgrid(edgesX._data[:-1], edgesY._data[:-1], indexing="ij")
        x1, y1 = meshgrid(edgesX._data[1:], edgesY._data[1:], indexing="ij")
        X0, X1 = Array("X0", x0), Array("X1", x1)
        Y0, Y1 = Array("Y0", y0), Array("Y1", y1)
        sampler = IntegratorSampler("sampler", mode="2d")
        integrator = IntegratorCore(
            "integrator",
            dropdim=dropdim,
            label={
                "plottitle": f"IntegratorCore test: {testname}",
                "axis": "integral",
            },
        )
        poly0 = Polynomial21("poly0")
        polyres = PolynomialRes("polyres")
        ordersX >> sampler("ordersX")
        ordersY >> sampler("ordersY")
        sampler.outputs["x"] >> poly0
        sampler.outputs["y"] >> poly0
        X0 >> polyres
        X1 >> polyres
        Y0 >> polyres
        Y1 >> polyres
        sampler.outputs["weights"] >> integrator("weights")
        poly0.outputs[0] >> integrator
        ordersX >> integrator("ordersX")
        ordersY >> integrator("ordersY")
    if dropdim:
        res = (polyres.outputs[1].data.T - polyres.outputs[0].data.T) * (
            polyres.outputs[3].data.T - polyres.outputs[2].data.T
        )[0]
        edges = [edgesX["array"]]
    else:
        res = (polyres.outputs[1].data - polyres.outputs[0].data) * (
            polyres.outputs[3].data - polyres.outputs[2].data
        )
        edges = [edgesX["array"], edgesY["array"]]
    assert allclose(integrator.outputs[0].data, res, atol=1e-10)
    assert integrator.outputs[0].dd.axes_edges == edges

    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("dropdim", (True, False))
def test_IntegratorCore_gl2to1d_y(debug_graph, testname, dropdim):
    class Polynomial21(ManyToOneNode):
        def _fcn(self):
            self.outputs["result"].data[:] = vecF0(self.inputs[1].data) * vecF0(self.inputs[0].data)

    with Graph(debug=debug_graph, close_on_exit=True) as graph:
        npointsY = 20
        edgesX = Array("edgesX", [0, 1], label={"axis": "Label for axis X"})
        edgesY = Array(
            "edgesY",
            linspace(2, 12, npointsY + 1),
            label={"axis": "Label for axis Y"},
        )
        ordersX = Array("ordersX", [2], edges=edgesX["array"])
        ordersY = Array("ordersY", [2] * npointsY, edges=edgesY["array"])
        x0, y0 = meshgrid(edgesX._data[:-1], edgesY._data[:-1], indexing="ij")
        x1, y1 = meshgrid(edgesX._data[1:], edgesY._data[1:], indexing="ij")
        X0, X1 = Array("X0", x0), Array("X1", x1)
        Y0, Y1 = Array("Y0", y0), Array("Y1", y1)
        sampler = IntegratorSampler("sampler", mode="2d")
        integrator = IntegratorCore(
            "integrator",
            dropdim=dropdim,
            label={
                "plottitle": f"IntegratorCore test: {testname}",
                "axis": "integral",
            },
        )
        poly0 = Polynomial21("poly0")
        polyres = PolynomialRes("polyres")
        ordersX >> sampler("ordersX")
        ordersY >> sampler("ordersY")
        sampler.outputs["x"] >> poly0
        sampler.outputs["y"] >> poly0
        X0 >> polyres
        X1 >> polyres
        Y0 >> polyres
        Y1 >> polyres
        sampler.outputs["weights"] >> integrator("weights")
        poly0.outputs[0] >> integrator
        ordersX >> integrator("ordersX")
        ordersY >> integrator("ordersY")
    if dropdim:
        res = (polyres.outputs[1].data - polyres.outputs[0].data) * (
            polyres.outputs[3].data - polyres.outputs[2].data
        )[0]
        edges = [edgesY["array"]]
    else:
        res = (polyres.outputs[1].data - polyres.outputs[0].data) * (
            polyres.outputs[3].data - polyres.outputs[2].data
        )
        edges = [edgesX["array"], edgesY["array"]]
    assert allclose(integrator.outputs[0].data, res, atol=1e-10)
    assert integrator.outputs[0].dd.axes_edges == edges

    savegraph(graph, f"output/{testname}.png")


def test_IntegratorCore_edges_0(debug_graph):
    """test wrong ordersX: edges not given"""
    arr = [1.0, 2.0, 3.0]
    with Graph(debug=debug_graph):
        arr1 = Array("array", arr)
        weights = Array("weights", arr)
        ordersX = Array("ordersX", [1, 2, 3])
        integrator = IntegratorCore("integrator")
        arr1 >> integrator
        weights >> integrator("weights")
        ordersX >> integrator("ordersX")
    with raises(TypeFunctionError):
        integrator.close()


def test_IntegratorCore_edges_1(debug_graph):
    """test wrong ordersX: edges is wrong"""
    arr = [1.0, 2.0, 3.0]
    with Graph(debug=debug_graph, close_on_exit=False):
        edges = Array("edges", [0.0, 1.0, 2.0])
        with raises(TypeFunctionError):
            arr1 = Array("array", arr, edges=edges["array"])
        edges = Array("edges", [0.0, 1.0, 2.0, 3.0])
        arr1 = Array("array", arr, edges=edges["array"])
        weights = Array("weights", arr)
        ordersX = Array("ordersX", [1, 2, 3])
        integrator = IntegratorCore("integrator")
        arr1 >> integrator
        weights >> integrator("weights")
        ordersX >> integrator("ordersX")
    with raises(TypeFunctionError):
        integrator.close()


def test_IntegratorCore_02(debug_graph):
    """test wrong ordersX: sum(ordersX) != shape"""
    arr = [1.0, 2.0, 3.0]
    with Graph(debug=debug_graph):
        edges = Array("edges", [0.0, 1.0, 2.0, 3.0])
        arr1 = Array("array", arr, edges=edges["array"])
        weights = Array("weights", arr, edges=edges["array"])
        ordersX = Array("ordersX", [1, 2, 3], edges=edges["array"])
        integrator = IntegratorCore("integrator")
        arr1 >> integrator
        weights >> integrator("weights")
        ordersX >> integrator("ordersX")
    with raises(TypeFunctionError):
        integrator.close()


def test_IntegratorCore_03(debug_graph):
    """test wrong ordersX: sum(ordersX[i]) != shape[i]"""
    arr = [1.0, 2.0, 3.0]
    with Graph(debug=debug_graph, close_on_exit=False):
        edgesX = Array("edgesX", [-1.0, 0.0, 1.0])
        edgesY = Array("edgesY", [-2.0, -1, 0.0, 1.0])
        arr1 = Array("array", [arr, arr], edges=[edgesX["array"], edgesY["array"]])
        weights = Array("weights", [arr, arr], edges=[edgesX["array"], edgesY["array"]])
        ordersX = Array("ordersX", [1, 3], edges=edgesX["array"])
        ordersY = Array("ordersY", [1, 0, 0], edges=edgesY["array"])
        integrator = IntegratorCore("integrator")
        arr1 >> integrator
        weights >> integrator("weights")
        ordersX >> integrator("ordersX")
        ordersY >> integrator("ordersY")
    with raises(TypeFunctionError):
        integrator.close()


def test_IntegratorCore_04(debug_graph):
    """test wrong shape"""
    with Graph(debug=debug_graph, close_on_exit=False):
        arr1 = Array("array", [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        arr2 = Array("array", [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
        weights = Array("weights", [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        ordersX = Array("ordersX", [0, 2])
        ordersY = Array("ordersY", [1, 1, 1, 3])
        integrator = IntegratorCore("integrator")
        arr1 >> integrator
        arr2 >> integrator
        weights >> integrator("weights")
        ordersX >> integrator("ordersX")
        ordersY >> integrator("ordersY")
    with raises(TypeFunctionError):
        integrator.close()
