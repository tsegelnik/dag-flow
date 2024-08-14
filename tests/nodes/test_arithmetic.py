#!/usr/bin/env python

import numpy
from numpy import allclose, arange, linspace, sum
from pytest import mark

from dagflow import lib
from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib import Array, Division, Product, Sum


@mark.parametrize("dtype", ("d", "f"))
def test_Sum_01(testname, debug_graph, dtype):
    arrays_in = tuple(arange(12, dtype=dtype) * i for i in (1, 2, 3))

    with Graph(close=True, debug=debug_graph) as graph:
        arrays = tuple(Array(f"arr_{i}", array_in) for i, array_in in enumerate(arrays_in))
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
        arrays = tuple(Array(f"arr_{i}", array_in) for i, array_in in enumerate(arrays_in))
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
        arrays = tuple(Array(f"arr_{i}", array_in) for i, array_in in enumerate(arrays_in))
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


@mark.parametrize("dtype", ("d", "f"))
@mark.parametrize("fcnname", ("square", "sqrt"))
def test_Powers_01(testname, debug_graph, fcnname, dtype):
    fcn_np = getattr(numpy, fcnname)
    fcn_node = getattr(lib, fcnname.capitalize())
    if fcnname in ("square"):
        arrays_in = tuple(linspace(-10, 10, 101, dtype=dtype) * i for i in (1, 2, 3))
    else:
        arrays_in = tuple(linspace(0, 10, 101, dtype=dtype) * i for i in (1, 2, 3))

    with Graph(close=True, debug=debug_graph) as graph:
        arrays = tuple(
            Array(f"arr_{i}", array_in, label={"text": f"X axis {i}"})
            for i, array_in in enumerate(arrays_in)
        )
        node = fcn_node(fcnname)
        arrays >> node

    outputs = node.outputs
    ress = fcn_np(arrays_in)

    assert node.tainted == True
    assert all(output.dd.dtype == dtype for output in outputs)
    assert allclose(tuple(outputs.iter_data()), ress, rtol=0, atol=0)
    assert node.tainted == False

    savegraph(graph, f"output/{testname}.png", show="full")
