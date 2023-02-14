#!/usr/bin/env python
from dagflow.exception import TypeFunctionError
from dagflow.graph import Graph
from dagflow.lib.Array import Array
from dagflow.lib.Integrator import Integrator

from pytest import raises


def test_Integrator_00(debug_graph):
    with Graph(debug=debug_graph, close=True):
        arr1 = Array("array", [1.0, 2.0, 3.0])
        arr2 = Array("array", [3.0, 2.0, 1.0])
        weights = Array("weights", [2.0, 2.0, 2.0])
        ordersX = Array("ordersX", [1, 1, 1])
        integrator = Integrator("integrator", mode="1d")
        arr1 >> integrator
        arr2 >> integrator
        weights >> integrator("weights")
        ordersX >> integrator("ordersX")
    assert (integrator.outputs[0].data == [2, 4, 6]).all()
    assert (integrator.outputs[1].data == [6, 4, 2]).all()


def test_Integrator_01(debug_graph):
    with Graph(debug=debug_graph, close=True):
        arr1 = Array("array", [1.0, 2.0, 3.0])
        arr2 = Array("array", [3.0, 2.0, 1.0])
        weights = Array("weights", [2.0, 2.0, 2.0])
        ordersX = Array("ordersX", [2, 0, 1])
        integrator = Integrator("integrator", mode="1d")
        arr1 >> integrator
        arr2 >> integrator
        weights >> integrator("weights")
        ordersX >> integrator("ordersX")
    assert (integrator.outputs[0].data == [6, 0, 6]).all()
    assert (integrator.outputs[1].data == [10, 0, 2]).all()


def test_Integrator_02(debug_graph):
    arr123 = [1.0, 2.0, 3.0]
    with Graph(debug=debug_graph, close=True):
        arr1 = Array("array", [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        arr2 = Array("array", [arr123, arr123])
        weights = Array("weights", [[1.0, 1.0, 1.0], arr123])
        ordersX = Array("ordersX", [1, 1])
        ordersY = Array("ordersY", [1, 1, 1])
        integrator = Integrator("integrator", mode="2d")
        arr1 >> integrator
        arr2 >> integrator
        weights >> integrator("weights")
        ordersX >> integrator("ordersX")
        ordersY >> integrator("ordersY")
    assert (integrator.outputs[0].data == [[1, 1, 1], [1, 2, 3]]).all()
    assert (integrator.outputs[1].data == [[1, 2, 3], [1, 4, 9]]).all()


def test_Integrator_03(debug_graph):
    arr123 = [1.0, 2.0, 3.0]
    with Graph(debug=debug_graph, close=True):
        arr1 = Array("array", [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        arr2 = Array("array", [arr123, arr123])
        weights = Array("weights", [[1.0, 1.0, 1.0], arr123])
        ordersX = Array("ordersX", [1, 1])
        ordersY = Array("ordersY", [1, 2, 0])
        integrator = Integrator("integrator", mode="2d")
        arr1 >> integrator
        arr2 >> integrator
        weights >> integrator("weights")
        ordersX >> integrator("ordersX")
        ordersY >> integrator("ordersY")
    assert (integrator.outputs[0].data == [[1, 2, 0], [1, 5, 0]]).all()
    assert (integrator.outputs[1].data == [[1, 5, 0], [1, 13, 0]]).all()


def test_Integrator_04(debug_graph):
    arr123 = [1.0, 2.0, 3.0]
    with Graph(debug=debug_graph, close=True):
        arr1 = Array("array", [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        arr2 = Array("array", [arr123, arr123])
        weights = Array("weights", [[1.0, 1.0, 1.0], arr123])
        ordersX = Array("ordersX", [0, 2])
        ordersY = Array("ordersY", [1, 1, 1])
        integrator = Integrator("integrator", mode="2d")
        arr1 >> integrator
        arr2 >> integrator
        weights >> integrator("weights")
        ordersX >> integrator("ordersX")
        ordersY >> integrator("ordersY")
    assert (integrator.outputs[0].data == [[0, 0, 0], [2, 3, 4]]).all()
    assert (integrator.outputs[1].data == [[0, 0, 0], [2, 6, 12]]).all()


def test_Integrator_05(debug_graph):
    unity = [1.0, 1.0, 1.0]
    with Graph(debug=debug_graph, close=True):
        arr1 = Array(
            "array", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        arr2 = Array("array", [unity, unity, unity])
        weights = Array("weights", [unity, unity, unity])
        ordersX = Array("ordersX", [1, 1, 1])
        ordersY = Array("ordersY", [1, 0, 2])
        integrator = Integrator("integrator", mode="2d")
        arr1 >> integrator
        arr2 >> integrator
        weights >> integrator("weights")
        ordersX >> integrator("ordersX")
        ordersY >> integrator("ordersY")
    assert (
        integrator.outputs[0].data == [[1, 0, 0], [0, 0, 1], [0, 0, 1]]
    ).all()
    assert (
        integrator.outputs[1].data == [[1, 0, 2], [1, 0, 2], [1, 0, 2]]
    ).all()


# test wrong ordersX: sum(ordersX) != shape
def test_Integrator_06(debug_graph):
    arr = [1.0, 2.0, 3.0]
    with Graph(debug=debug_graph):
        arr1 = Array("array", arr)
        weights = Array("weights", arr)
        ordersX = Array("ordersX", [1, 2, 3])
        integrator = Integrator("integrator", mode="1d")
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
        integrator = Integrator("integrator", mode="2d")
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
        integrator = Integrator("integrator", mode="2d")
        arr1 >> integrator
        arr2 >> integrator
        weights >> integrator("weights")
        ordersX >> integrator("ordersX")
        ordersY >> integrator("ordersY")
    with raises(TypeFunctionError):
        integrator.close()
