#!/usr/bin/env python

from dagflow.graph import Graph
from dagflow.lib import Array, Product, Sum, WeightedSum
from numpy import arange, array


def test_00(debug_graph):
    with Graph(debug=debug_graph, close=True):
        arr = Array("arr", arange(3, dtype="d"))  # [0, 1, 2]
        ws = WeightedSum("weightedsum")
        (arr, arr) >> ws
        Array("weight", (2, 3)) >> ws("weight")
    assert ws.closed
    assert (ws.outputs["result"].data == [0, 5, 10]).all()
    assert arr.open()
    assert not ws.inputs["weight"].closed
    assert not arr.closed


def test_01(debug_graph):
    with Graph(debug=debug_graph, close=True):
        arr1 = Array("arr1", arange(3, dtype="d"))  # [0, 1, 2]
        arr2 = Array("arr2", array((3, 2, 1), dtype="d"))
        sum = Sum("sum")
        (arr1, arr2) >> sum
    assert sum.closed
    assert (sum.outputs["result"].data == [3, 3, 3]).all()
    assert sum.open()
    assert all((not sum.closed, arr1.closed, arr2.closed))
    assert arr1.open()
    assert all((not sum.closed, not arr1.closed, arr2.closed))
    assert arr2.open()
    assert all((not sum.closed, not arr1.closed, not arr2.closed))


def test_02(debug_graph):
    with Graph(debug=debug_graph, close=True):
        arr1 = Array("arr1", arange(3, dtype="d"))  # [0, 1, 2]
        arr2 = Array("arr2", array((3, 2, 1), dtype="d"))
        arr3 = Array("unity", array((1, 1, 1), dtype="d"))
        sum1 = Sum("sum1")
        sum2 = Sum("sum2")
        prod = Product("product")
        (arr1, arr2, arr3) >> sum1  # [4, 4, 4]
        (arr3, sum1) >> prod  # [4, 4, 4]
        (arr1, prod) >> sum2  # [4, 5, 6]
    assert sum2.closed
    assert (sum2.outputs["result"].data == [4, 5, 6]).all()
    assert arr1.open()
    assert arr2.closed
    assert arr3.closed
    assert not arr1.closed
    assert not prod.closed
    assert not sum1.closed


def test_03(debug_graph):
    with Graph(debug=debug_graph, close=False):
        arr1 = Array("arr1", arange(3, dtype="d"))  # [0, 1, 2]
        arr2 = Array("arr2", array((3, 2, 1), dtype="d"))
        arr3 = Array("unity", array((1, 1, 1), dtype="d"))
        sum1 = Sum("sum1")
        sum2 = Sum("sum2")
        prod = Product("product")
        (arr1, arr2, arr3) >> sum1  # [4, 4, 4]
        (arr3, sum1) >> prod  # [4, 4, 4]
        (arr1, prod) >> sum2  # [4, 5, 6]

    with Graph(debug=debug_graph, close=True):
        arr4 = Array("arr1", arange(3, dtype="d"))  # [0, 1, 2]
        sum3 = Sum("sum3")
        (sum2, arr4) >> sum3  # [4, 7, 8]
    assert arr1.closed
    assert arr2.closed
    assert arr3.closed
    assert arr4.closed
    assert sum2.closed
    assert sum3.closed
    assert (sum3.outputs["result"].data == [4, 6, 8]).all()
    assert arr1.open()
    assert arr2.closed
    assert arr3.closed
    assert arr4.closed
    assert not arr1.closed
    assert not prod.closed
    assert not sum1.closed
    assert not sum2.closed
