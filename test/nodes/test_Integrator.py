#!/usr/bin/env python
from dagflow.graph import Graph
from dagflow.lib.Array import Array
from dagflow.lib.Integrator import Integrator


def test_Integrator_00(debug_graph):
    with Graph(debug=debug_graph, close=True):
        arr1 = Array("array", [1, 2, 3])
        arr2 = Array("array", [3, 2, 1])
        weights = Array("weights", [1, 1, 1])
        orders = Array("orders", [1, 0, 1])
        integrator = Integrator("integrator")
        arr1 >> integrator
        arr2 >> integrator
        weights >> integrator("weights")
        orders >> integrator("orders")
    assert all(integrator.outputs[0].data == [1, 0, 2])
    assert all(integrator.outputs[1].data == [3, 0, 2])


def test_Integrator_01(debug_graph):
    unity = [1, 1, 1]
    with Graph(debug=debug_graph, close=True):
        arr1 = Array("array", [unity, unity])
        arr2 = Array("array", [[1, 2, 3], [1, 2, 3]])
        weights = Array("weights", [unity, unity])
        orders = Array("orders", [[1, 1, 0], unity])
        integrator = Integrator("integrator")
        arr1 >> integrator
        arr2 >> integrator
        weights >> integrator("weights")
        orders >> integrator("orders")
    assert (integrator.outputs[0].data == [[1, 1, 1], [1, 1, 1]]).all()
    assert (integrator.outputs[1].data == [[1, 2, 3], [1, 2, 3]]).all()


def test_Integrator_02(debug_graph):
    unity = [1, 1, 1]
    with Graph(debug=debug_graph, close=True):
        arr1 = Array("array", [unity, unity])
        arr2 = Array("array", [[1, 2, 3], [1, 2, 3]])
        weights = Array("weights", [[0, 1, 2], [3, 4, 5]])
        orders = Array("orders", [[0, 2, 0], [2, 1, 0]])
        integrator = Integrator("integrator")
        arr1 >> integrator
        arr2 >> integrator
        weights >> integrator("weights")
        orders >> integrator("orders")
    assert (integrator.outputs[0].data == [[0, 0, 0], [8, 7, 0]]).all()
    assert (integrator.outputs[1].data == [[0, 0, 0], [13, 21, 0]]).all()
