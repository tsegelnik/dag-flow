#!/usr/bin/env python
from numpy import arange
from numpy import sum
from pytest import mark

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib import Array
from dagflow.lib import Division
from dagflow.lib import Product
from dagflow.lib import Sum


@mark.parametrize("dtype", ("d", "f"))
def test_Sum_01(testname, debug_graph, dtype):
    arrays_in = tuple(arange(12, dtype=dtype) * i for i in (1, 2, 3))

    with Graph(close=True, debug=debug_graph) as graph:
        arrays = tuple(
            Array(f"arr_{i}", array_in) for i, array_in in enumerate(arrays_in)
        )
        sm = Sum("sum")
        arrays >> sm

    output = sm.outputs[0]

    res = sum(arrays_in, axis=0)

    assert sm.tainted == True
    assert all(output.data == res)
    assert sm.tainted == False

    arrays_in = (arrays_in[1],) + arrays_in[1:]
    res = sum(arrays_in, axis=0)
    assert arrays[0].set(arrays[1].get_data())
    assert sm.tainted == True
    assert all(output.data == res)
    assert sm.tainted == False

    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("dtype", ("d", "f"))
def test_Product_01(testname, debug_graph, dtype):
    arrays_in = tuple(arange(12, dtype=dtype) * i for i in (1, 2, 3))

    with Graph(close=True, debug=debug_graph) as graph:
        arrays = tuple(
            Array(f"arr_{i}", array_in) for i, array_in in enumerate(arrays_in)
        )
        prod = Product("prod")
        arrays >> prod

    output = prod.outputs[0]
    res = arrays_in[0] * arrays_in[1] * arrays_in[2]

    assert prod.tainted == True
    assert (output.data == res).all()
    assert prod.tainted == False

    arrays_in = (arrays_in[1],) + arrays_in[1:]
    res = arrays_in[0] * arrays_in[1] * arrays_in[2]
    assert arrays[0].set(arrays[1].get_data())
    assert prod.tainted == True
    assert all(output.data == res)
    assert prod.tainted == False

    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("dtype", ("d", "f"))
def test_Division_01(testname, debug_graph, dtype):
    arrays_in = tuple(arange(12, dtype=dtype) * i + 1 for i in (1, 2, 3))

    with Graph(close=True, debug=debug_graph) as graph:
        arrays = tuple(
            Array(f"arr_{i}", array_in) for i, array_in in enumerate(arrays_in)
        )
        div = Division("division")
        arrays >> div

    output = div.outputs[0]
    res = arrays_in[0] / arrays_in[1] / arrays_in[2]

    assert div.tainted == True
    assert (output.data == res).all()
    assert div.tainted == False

    arrays_in = (arrays_in[1],) + arrays_in[1:]
    res = arrays_in[0] / arrays_in[1] / arrays_in[2]
    assert arrays[0].set(arrays[1].get_data())
    assert div.tainted == True
    assert all(output.data == res)
    assert div.tainted == False

    savegraph(graph, f"output/{testname}.png")
