#!/usr/bin/env python

from dagflow.graph import Graph
from dagflow.lib.Array import Array
from dagflow.lib.trigonometry import Cos, Sin, Tan, ArcCos, ArcSin, ArcTan # Accessed via globals()
from dagflow.graphviz import savegraph
from dagflow.plot import plot_auto

from numpy import allclose, pi, linspace
from numpy import cos, sin, tan, arccos, arcsin, arctan # accessed via globals()
from matplotlib.pyplot import close
from pytest import mark

@mark.parametrize("dtype", ("d", "f"))
@mark.parametrize("fcnname", ("cos", "sin", "tan", "arccos", "arcsin", "arctan"))
def test_Cos_01(testname, debug_graph, fcnname, dtype):
    fcn_np = globals()[fcnname]
    fcn_node = globals()[f"{fcnname[:3].capitalize()}{fcnname[3:].capitalize()}"]
    if fcnname in ("cos", "sin", "tan"):
        arrays_in = tuple(linspace(-2*pi, 2*pi, 101, dtype=dtype) * i for i in (1, 2, 3))
    elif fcnname=="arctan":
        arrays_in = tuple(linspace(-10, 10, 101, dtype=dtype) * i for i in (1, 2, 3))
    else:
        arrays_in = tuple(linspace(-1, 1, 101, dtype=dtype)/i for i in (1, 2, 3))

    with Graph(close=True, debug=debug_graph) as graph:
        arrays = tuple(
            Array(f"arr_{i}", array_in, label={
                'text': f'X axis {i}'
            }) for i, array_in in enumerate(arrays_in)
        )
        node = fcn_node(fcnname)
        arrays >> node

    outputs = node.outputs
    ress = fcn_np(arrays_in)

    assert node.tainted == True
    assert all(output.dd.dtype == dtype for output in outputs)
    assert allclose(tuple(outputs.iter_data()), ress, rtol=0, atol=0)
    assert node.tainted == False

    plot_auto(node.outputs[0], label='input 0')
    plot_auto(node.outputs[1], label='input 1')
    plot_auto(node.outputs[2], label='input 2')
    close()

    savegraph(graph, f"output/{testname}.png")
