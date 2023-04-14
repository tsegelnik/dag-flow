#!/usr/bin/env python
from dagflow.exception import TypeFunctionError
from dagflow.graph import Graph
from dagflow.lib.Array import Array
from dagflow.lib.Integrator import Integrator
from dagflow.lib.IntegratorSampler import IntegratorSampler
from dagflow.lib.N2One import N2One
from dagflow.lib.One2One import One2One
from dagflow.lib.trigonometry import Cos, Sin
from numpy import allclose, linspace, meshgrid, pi, vectorize
from pytest import mark, raises


@mark.parametrize("align", ("left", "center", "right"))
def test_Integrator_rect_center(align, debug_graph):
    with Graph(debug=debug_graph, close=True):
        npoints = 10
        ordersX = Array("ordersX", [1000] * npoints)
        edges = linspace(0, pi, npoints + 1)
        ordersX.outputs[0].dd.axes_edges = edges
        A = Array("A", edges[:-1])
        B = Array("B", edges[1:])
        sampler = IntegratorSampler("sampler", mode="rect", align=align)
        integrator = Integrator("integrator")
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


def test_Integrator_trap(debug_graph):
    with Graph(debug=debug_graph, close=True):
        npoints = 10
        ordersX = Array("ordersX", [1000] * npoints)
        edges = linspace(0, pi, npoints + 1)
        ordersX.outputs[0].dd.axes_edges = edges
        A = Array("A", edges[:-1])
        B = Array("B", edges[1:])
        sampler = IntegratorSampler("sampler", mode="trap")
        integrator = Integrator("integrator")
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
    # TODO: why is there the very bad accuracy?
    assert allclose(integrator.outputs[0].data, res, atol=1e-2)


def f0(x: float) -> float:
    return 4 * x**3 + 3 * x**2 + 2 * x - 1


def fres(x: float) -> float:
    return x**4 + x**3 + x**2 - x


vecF0 = vectorize(f0)
vecFres = vectorize(fres)


class Polynomial0(One2One):
    def _fcn(self, _, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            out.data[:] = vecF0(inp.data)
        return list(outputs.iter_data())


class PolynomialRes(One2One):
    def _fcn(self, _, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            out.data[:] = vecFres(inp.data)
        return list(outputs.iter_data())


def test_Integrator_gl1d(debug_graph):
    with Graph(debug=debug_graph, close=True):
        npoints = 10
        ordersX = Array("ordersX", [2] * npoints)
        edges = linspace(0, 10, npoints + 1)
        ordersX.outputs[0].dd.axes_edges = edges
        A = Array("A", edges[:-1])
        B = Array("B", edges[1:])
        sampler = IntegratorSampler("sampler", mode="gl")
        integrator = Integrator("integrator")
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


def test_Integrator_gl2d(debug_graph):
    class Polynomial1(N2One):
        def _fcn(self, _, inputs, outputs):
            outputs["result"].data[:] = vecF0(inputs[1].data) * vecF0(
                inputs[0].data
            )
            return list(outputs.iter_data())

    with Graph(debug=debug_graph, close=True):
        npointsX, npointsY = 10, 20
        ordersX = Array("ordersX", [2] * npointsX)
        ordersY = Array("ordersY", [2] * npointsY)
        edgesX = linspace(0, 10, npointsX + 1)
        edgesY = linspace(0, 10, npointsY + 1)
        ordersX.outputs[0].dd.axes_edges = edgesX
        ordersY.outputs[0].dd.axes_edges = edgesY
        x0, y0 = meshgrid(edgesX[:-1], edgesY[:-1], indexing="ij")
        x1, y1 = meshgrid(edgesX[1:], edgesY[1:], indexing="ij")
        X0, X1 = Array("X0", x0), Array("X1", x1)
        Y0, Y1 = Array("Y0", y0), Array("Y1", y1)
        sampler = IntegratorSampler("sampler", mode="2d")
        integrator = Integrator("integrator")
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


# test wrong ordersX: sum(ordersX) != shape
def test_Integrator_01(debug_graph):
    arr = [1.0, 2.0, 3.0]
    with Graph(debug=debug_graph):
        arr1 = Array("array", arr)
        weights = Array("weights", arr)
        ordersX = Array("ordersX", [1, 2, 3])
        integrator = Integrator("integrator")
        arr1 >> integrator
        weights >> integrator("weights")
        ordersX >> integrator("ordersX")
    with raises(TypeFunctionError):
        integrator.close()


# test wrong ordersX: sum(ordersX[i]) != shape[i]
def test_Integrator_02(debug_graph):
    arr = [1.0, 2.0, 3.0]
    with Graph(debug=debug_graph):
        arr1 = Array("array", [arr, arr])
        weights = Array("weights", [arr, arr])
        ordersX = Array("ordersX", [1, 3])
        ordersY = Array("ordersY", [1, 0, 0])
        integrator = Integrator("integrator")
        arr1 >> integrator
        weights >> integrator("weights")
        ordersX >> integrator("ordersX")
        ordersY >> integrator("ordersY")
    with raises(TypeFunctionError):
        integrator.close()


# test wrong shape
def test_Integrator_03(debug_graph):
    with Graph(debug=debug_graph, close=False):
        arr1 = Array("array", [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        arr2 = Array("array", [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
        weights = Array("weights", [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        ordersX = Array("ordersX", [0, 2])
        ordersY = Array("ordersY", [1, 1, 1, 3])
        integrator = Integrator("integrator")
        arr1 >> integrator
        arr2 >> integrator
        weights >> integrator("weights")
        ordersX >> integrator("ordersX")
        ordersY >> integrator("ordersY")
    with raises(TypeFunctionError):
        integrator.close()
