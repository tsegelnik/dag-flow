#!/usr/bin/env python
from dagflow.exception import TypeFunctionError
from dagflow.graph import Graph
from dagflow.lib.Array import Array
from dagflow.lib.Integrator import Integrator
from dagflow.lib.IntegratorSampler import IntegratorSampler
from dagflow.lib.NodeManyToOne import NodeManyToOne
from dagflow.lib.NodeOneToOne import NodeOneToOne
from dagflow.lib.trigonometry import Cos, Sin
from numpy import allclose, linspace, meshgrid, pi, vectorize
from pytest import mark, raises


@mark.parametrize("align", ("left", "center", "right"))
def test_Integrator_rect_center(align, debug_graph):
    with Graph(debug=debug_graph, close=True):
        npoints = 10
        edges = Array("edges", linspace(0, pi, npoints + 1))
        ordersX = Array("ordersX", [1000] * npoints, edges=edges["array"])
        A = Array("A", edges._data[:-1])
        B = Array("B", edges._data[1:])
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
    assert integrator.outputs[0].dd.axes_edges == [edges["array"]]


def test_Integrator_trap(debug_graph):
    with Graph(debug=debug_graph, close=True):
        npoints = 10
        edges = Array("edges", linspace(0, pi, npoints + 1))
        ordersX = Array("ordersX", [1000] * npoints, edges=edges["array"])
        A = Array("A", edges._data[:-1])
        B = Array("B", edges._data[1:])
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
    assert allclose(integrator.outputs[0].data, res, atol=1e-2)
    assert integrator.outputs[0].dd.axes_edges == [edges["array"]]


def f0(x: float) -> float:
    return 4 * x**3 + 3 * x**2 + 2 * x - 1


def fres(x: float) -> float:
    return x**4 + x**3 + x**2 - x


vecF0 = vectorize(f0)
vecFres = vectorize(fres)


class Polynomial0(NodeOneToOne):
    def _fcn(self, _, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            out.data[:] = vecF0(inp.data)
        return list(outputs.iter_data())


class PolynomialRes(NodeOneToOne):
    def _fcn(self, _, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            out.data[:] = vecFres(inp.data)
        return list(outputs.iter_data())


def test_Integrator_gl1d(debug_graph):
    with Graph(debug=debug_graph, close=True):
        npoints = 10
        edges = Array("edges", linspace(0, 10, npoints + 1))
        ordersX = Array("ordersX", [2] * npoints, edges=edges["array"])
        A = Array("A", edges._data[:-1])
        B = Array("B", edges._data[1:])
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
    assert integrator.outputs[0].dd.axes_edges == [edges["array"]]


def test_Integrator_gl2d(debug_graph):
    class Polynomial1(NodeManyToOne):
        def _fcn(self, _, inputs, outputs):
            outputs["result"].data[:] = vecF0(inputs[1].data) * vecF0(
                inputs[0].data
            )
            return list(outputs.iter_data())

    with Graph(debug=debug_graph, close=True):
        npointsX, npointsY = 10, 20
        edgesX = Array("edgesX", linspace(0, 10, npointsX + 1))
        edgesY = Array("edgesY", linspace(0, 10, npointsY + 1))
        ordersX = Array("ordersX", [2] * npointsX, edges=edgesX["array"])
        ordersY = Array("ordersY", [2] * npointsY, edges=edgesY["array"])
        x0, y0 = meshgrid(edgesX._data[:-1], edgesY._data[:-1], indexing="ij")
        x1, y1 = meshgrid(edgesX._data[1:], edgesY._data[1:], indexing="ij")
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
    assert integrator.outputs[0].dd.axes_edges == [
        edgesX["array"],
        edgesY["array"],
    ]


# test wrong ordersX: edges not given
def test_Integrator_edges_0(debug_graph):
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


# test wrong ordersX: edges is wrong
def test_Integrator_edges_1(debug_graph):
    arr = [1.0, 2.0, 3.0]
    with Graph(debug=debug_graph, close=False):
        edges = Array("edges", [0.0, 1.0, 2.0])
        with raises(TypeFunctionError):
            arr1 = Array("array", arr, edges=edges["array"])
        edges = Array("edges", [0.0, 1.0, 2.0, 3.0])
        arr1 = Array("array", arr, edges=edges["array"])
        weights = Array("weights", arr)
        ordersX = Array("ordersX", [1, 2, 3])
        integrator = Integrator("integrator")
        arr1 >> integrator
        weights >> integrator("weights")
        ordersX >> integrator("ordersX")
    with raises(TypeFunctionError):
        integrator.close()


# test wrong ordersX: sum(ordersX) != shape
def test_Integrator_02(debug_graph):
    arr = [1.0, 2.0, 3.0]
    with Graph(debug=debug_graph):
        edges = Array("edges", [0.0, 1.0, 2.0, 3.0])
        arr1 = Array("array", arr, edges=edges["array"])
        weights = Array("weights", arr, edges=edges["array"])
        ordersX = Array("ordersX", [1, 2, 3], edges=edges["array"])
        integrator = Integrator("integrator")
        arr1 >> integrator
        weights >> integrator("weights")
        ordersX >> integrator("ordersX")
    with raises(TypeFunctionError):
        integrator.close()


# test wrong ordersX: sum(ordersX[i]) != shape[i]
def test_Integrator_03(debug_graph):
    arr = [1.0, 2.0, 3.0]
    with Graph(debug=debug_graph, close=False):
        edgesX = Array("edgesX", [-1.0, 0.0, 1.0])
        edgesY = Array("edgesY", [-2.0, -1, 0.0, 1.0])
        arr1 = Array(
            "array", [arr, arr], edges=[edgesX["array"], edgesY["array"]]
        )
        weights = Array(
            "weights", [arr, arr], edges=[edgesX["array"], edgesY["array"]]
        )
        ordersX = Array("ordersX", [1, 3], edges=edgesX["array"])
        ordersY = Array("ordersY", [1, 0, 0], edges=edgesY["array"])
        integrator = Integrator("integrator")
        arr1 >> integrator
        weights >> integrator("weights")
        ordersX >> integrator("ordersX")
        ordersY >> integrator("ordersY")
    with raises(TypeFunctionError):
        integrator.close()


# test wrong shape
def test_Integrator_04(debug_graph):
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
