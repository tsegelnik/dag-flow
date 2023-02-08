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
        weights = Array("weights", [1.0, 1.0, 1.0])
        orders = Array("orders", [1, 0, 1])
        integrator = Integrator("integrator")
        arr1 >> integrator
        arr2 >> integrator
        weights >> integrator("weights")
        orders >> integrator("orders")
    assert (integrator.outputs[0].data == [1, 2]).all()
    assert (integrator.outputs[1].data == [3, 2]).all()


def test_Integrator_01(debug_graph):
    arr123 = [1.0, 2.0, 3.0]
    with Graph(debug=debug_graph, close=True):
        arr1 = Array("array", [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        arr2 = Array("array", [arr123, arr123])
        weights = Array("weights", [[1.0, 1.0, 1.0], arr123])
        orders = Array("orders", [[1, 1, 0], [1, 1, 1]])
        integrator = Integrator("integrator")
        arr1 >> integrator
        arr2 >> integrator
        weights >> integrator("weights")
        orders >> integrator("orders")
    assert (integrator.outputs[0].data == [[1, 1, 1], [1, 2, 3]]).all()
    assert (integrator.outputs[1].data == [[1, 2, 3], [1, 4, 9]]).all()


def test_Integrator_02(debug_graph):
    arr123 = [1.0, 2.0, 3.0]
    with Graph(debug=debug_graph, close=True):
        arr1 = Array("array", [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        arr2 = Array("array", [arr123, arr123])
        weights = Array("weights", [[1.0, 1.0, 1.0], arr123])
        orders = Array("orders", [[1, 1, 0], [1, 0, 2]])
        integrator = Integrator("integrator")
        arr1 >> integrator
        arr2 >> integrator
        weights >> integrator("weights")
        orders >> integrator("orders")
    assert (integrator.outputs[0].data == [[1, 2], [1, 5]]).all()
    assert (integrator.outputs[1].data == [[1, 5], [1, 13]]).all()


def test_Integrator_03(debug_graph):
    arr123 = [1.0, 2.0, 3.0]
    with Graph(debug=debug_graph, close=True):
        arr1 = Array("array", [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        arr2 = Array("array", [arr123, arr123])
        weights = Array("weights", [[1.0, 1.0, 1.0], arr123])
        orders = Array("orders", [[1, 1, 0], [1, 0, 0]])
        integrator = Integrator("integrator")
        arr1 >> integrator
        arr2 >> integrator
        weights >> integrator("weights")
        orders >> integrator("orders")
    assert (integrator.outputs[0].data == [[1], [1]]).all()
    assert (integrator.outputs[1].data == [[1], [1]]).all()


def test_Integrator_04(debug_graph):
    arr123 = [1.0, 2.0, 3.0]
    with Graph(debug=debug_graph, close=True):
        arr1 = Array("array", [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        arr2 = Array("array", [arr123, arr123])
        weights = Array("weights", [[1.0, 1.0, 1.0], arr123])
        orders = Array("orders", [[1, 0, 0], [1, 2, 0]])
        integrator = Integrator("integrator")
        arr1 >> integrator
        arr2 >> integrator
        weights >> integrator("weights")
        orders >> integrator("orders")
    assert (integrator.outputs[0].data == [[1, 2]]).all()
    assert (integrator.outputs[1].data == [[1, 5]]).all()


def test_Integrator_05(debug_graph):
    arr = [1.0, 2.0, 3.0]
    with Graph(debug=debug_graph):
        arr1 = Array("array", arr)
        weights = Array("weights", arr)
        orders = Array("orders", [1, 2, 3])
        integrator = Integrator("integrator")
        arr1 >> integrator
        weights >> integrator("weights")
        orders >> integrator("orders")
    with raises(TypeFunctionError):
        integrator.close()


def test_Integrator_06(debug_graph):
    arr = [1.0, 2.0, 3.0]
    with Graph(debug=debug_graph):
        arr1 = Array("array", [arr, arr])
        weights = Array("weights", [arr, arr])
        orders = Array("orders", [[1, 3, 0], [1, 0, 0]])
        integrator = Integrator("integrator")
        arr1 >> integrator
        weights >> integrator("weights")
        orders >> integrator("orders")
    with raises(TypeFunctionError):
        integrator.close()
