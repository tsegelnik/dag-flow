#!/usr/bin/env python
from dagflow.exception import TypeFunctionError
from dagflow.graph import Graph
from dagflow.lib.Array import Array
from dagflow.lib.One2One import One2One
from dagflow.lib.trigonometry import Cos, Sin
from dagflow.lib.Integrator import Integrator
from dagflow.lib.IntegratorSampler import IntegratorSampler

from numpy import allclose, linspace, pi, vectorize
from pytest import raises


# TODO: implement left and right rect tests
def test_Integrator_rect_center(debug_graph):
    with Graph(debug=debug_graph, close=True):
        npoints = 10
        ordersX = Array("ordersX", [30] * npoints)
        edges = linspace(0, pi, npoints + 1)
        ordersX.outputs[0].dd.axes_edges = edges
        A = Array("X", edges[:-1])
        B = Array("X", edges[1:])
        sampler = IntegratorSampler("sampler", mode="rect", align="center")
        integrator = Integrator("integrator")
        cosf = Cos("cos")
        sinf = Sin("sin")
        ordersX >> sampler("ordersX")
        sampler.outputs["sample"] >> cosf
        A >> sinf
        B >> sinf
        sampler.outputs["weights"] >> integrator("weights")
        cosf.outputs[0] >> integrator
        ordersX >> integrator("ordersX")
    assert allclose(
        integrator.outputs[0].data,
        (sinf.outputs[1].data - sinf.outputs[0].data),
    )


def test_Integrator_trap(debug_graph):
    with Graph(debug=debug_graph, close=True):
        npoints = 10
        ordersX = Array("ordersX", [40] * npoints)
        edges = linspace(0, pi, npoints + 1)
        ordersX.outputs[0].dd.axes_edges = edges
        A = Array("X", edges[:-1])
        B = Array("X", edges[1:])
        sampler = IntegratorSampler("sampler", mode="trap")
        integrator = Integrator("integrator")
        cosf = Cos("cos")
        sinf = Sin("sin")
        ordersX >> sampler("ordersX")
        sampler.outputs["sample"] >> cosf
        A >> sinf
        B >> sinf
        sampler.outputs["weights"] >> integrator("weights")
        cosf.outputs[0] >> integrator
        ordersX >> integrator("ordersX")
    assert allclose(
        integrator.outputs[0].data,
        (sinf.outputs[1].data - sinf.outputs[0].data),
        atol=1e-1,
    )  # TODO: why is there the very bad accuracy?


def test_Integrator_gl_1(debug_graph):
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

    with Graph(debug=debug_graph, close=True):
        npoints = 10
        ordersX = Array("ordersX", [2] * npoints)
        edges = linspace(0, 10, npoints + 1)
        ordersX.outputs[0].dd.axes_edges = edges
        A = Array("X", edges[:-1])
        B = Array("X", edges[1:])
        sampler = IntegratorSampler("sampler", mode="gl")
        integrator = Integrator("integrator")
        poly0 = Polynomial0("poly0")
        polyres = PolynomialRes("polyres")
        ordersX >> sampler("ordersX")
        sampler.outputs["sample"] >> poly0
        A >> polyres
        B >> polyres
        sampler.outputs["weights"] >> integrator("weights")
        poly0.outputs[0] >> integrator
        ordersX >> integrator("ordersX")
    res = polyres.outputs[1].data - polyres.outputs[0].data
    assert allclose(integrator.outputs[0].data, res, atol=1e-10)


# TODO: fix old tests

# def test_Integrator_01(debug_graph):
#    with Graph(debug=debug_graph, close=True):
#        arr1 = Array("array", [1.0, 2.0, 3.0])
#        arr2 = Array("array", [3.0, 2.0, 1.0])
#        weights = Array("weights", [2.0, 2.0, 2.0])
#        ordersX = Array("ordersX", [2, 0, 1])
#        integrator = Integrator("integrator")
#        arr1 >> integrator
#        arr2 >> integrator
#        weights >> integrator("weights")
#        ordersX >> integrator("ordersX")
#    assert (integrator.outputs[0].data == [6, 0, 6]).all()
#    assert (integrator.outputs[1].data == [10, 0, 2]).all()
#
#
# def test_Integrator_02(debug_graph):
#    arr123 = [1.0, 2.0, 3.0]
#    with Graph(debug=debug_graph, close=True):
#        arr1 = Array("array", [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
#        arr2 = Array("array", [arr123, arr123])
#        weights = Array("weights", [[1.0, 1.0, 1.0], arr123])
#        ordersX = Array("ordersX", [1, 1])
#        ordersY = Array("ordersY", [1, 1, 1])
#        integrator = Integrator("integrator")
#        arr1 >> integrator
#        arr2 >> integrator
#        weights >> integrator("weights")
#        ordersX >> integrator("ordersX")
#        ordersY >> integrator("ordersY")
#    assert (integrator.outputs[0].data == [[1, 1, 1], [1, 2, 3]]).all()
#    assert (integrator.outputs[1].data == [[1, 2, 3], [1, 4, 9]]).all()
#
#
# def test_Integrator_03(debug_graph):
#    arr123 = [1.0, 2.0, 3.0]
#    with Graph(debug=debug_graph, close=True):
#        arr1 = Array("array", [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
#        arr2 = Array("array", [arr123, arr123])
#        weights = Array("weights", [[1.0, 1.0, 1.0], arr123])
#        ordersX = Array("ordersX", [1, 1])
#        ordersY = Array("ordersY", [1, 2, 0])
#        integrator = Integrator("integrator")
#        arr1 >> integrator
#        arr2 >> integrator
#        weights >> integrator("weights")
#        ordersX >> integrator("ordersX")
#        ordersY >> integrator("ordersY")
#    assert (integrator.outputs[0].data == [[1, 2, 0], [1, 5, 0]]).all()
#    assert (integrator.outputs[1].data == [[1, 5, 0], [1, 13, 0]]).all()
#
#
# def test_Integrator_04(debug_graph):
#    arr123 = [1.0, 2.0, 3.0]
#    with Graph(debug=debug_graph, close=True):
#        arr1 = Array("array", [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
#        arr2 = Array("array", [arr123, arr123])
#        weights = Array("weights", [[1.0, 1.0, 1.0], arr123])
#        ordersX = Array("ordersX", [0, 2])
#        ordersY = Array("ordersY", [1, 1, 1])
#        integrator = Integrator("integrator")
#        arr1 >> integrator
#        arr2 >> integrator
#        weights >> integrator("weights")
#        ordersX >> integrator("ordersX")
#        ordersY >> integrator("ordersY")
#    assert (integrator.outputs[0].data == [[0, 0, 0], [2, 3, 4]]).all()
#    assert (integrator.outputs[1].data == [[0, 0, 0], [2, 6, 12]]).all()
#
#
# def test_Integrator_05(debug_graph):
#    unity = [1.0, 1.0, 1.0]
#    with Graph(debug=debug_graph, close=True):
#        arr1 = Array(
#            "array", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
#        )
#        arr2 = Array("array", [unity, unity, unity])
#        weights = Array("weights", [unity, unity, unity])
#        ordersX = Array("ordersX", [1, 1, 1])
#        ordersY = Array("ordersY", [1, 0, 2])
#        integrator = Integrator("integrator")
#        arr1 >> integrator
#        arr2 >> integrator
#        weights >> integrator("weights")
#        ordersX >> integrator("ordersX")
#        ordersY >> integrator("ordersY")
#    assert (
#        integrator.outputs[0].data == [[1, 0, 0], [0, 0, 1], [0, 0, 1]]
#    ).all()
#    assert (
#        integrator.outputs[1].data == [[1, 0, 2], [1, 0, 2], [1, 0, 2]]
#    ).all()


# test wrong ordersX: sum(ordersX) != shape
def test_Integrator_06(debug_graph):
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
def test_Integrator_07(debug_graph):
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
def test_Integrator_08(debug_graph):
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
