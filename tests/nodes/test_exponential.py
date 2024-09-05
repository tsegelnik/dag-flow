import numpy
from matplotlib.pyplot import close
from numpy import allclose
from numpy import finfo
from numpy import linspace
from pytest import mark

from dagflow import lib
from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib import Array
from dagflow.plot import plot_auto


@mark.parametrize("dtype", ("d", "f"))
@mark.parametrize("fcnname", ("exp", "expm1", "log", "log1p", "log10"))
def test_Exponential_01(testname, debug_graph, fcnname, dtype):
    fcn_np = getattr(numpy, fcnname)
    fcn_node = getattr(lib, fcnname.capitalize())
    n = 101
    if "m1" in fcnname or "1p" in fcnname:
        arrays_in = tuple(linspace(-0.3, 0.3, n, dtype=dtype) * i for i in (1, 2, 3))
    elif fcnname.startswith("log"):
        arrays_in = tuple(linspace(0, 10, n + 1, dtype=dtype)[1:] * i for i in (1, 2, 3))
    elif fcnname == "exp":
        arrays_in = tuple(linspace(-10, 10, n, dtype=dtype) * i for i in (1, 2, 3))
    else:
        raise RuntimeError()

    with Graph(close_on_exit=True, debug=debug_graph) as graph:
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
    atol = finfo("d").resolution * 2
    assert allclose(tuple(outputs.iter_data()), ress, rtol=0, atol=atol)
    assert node.tainted == False

    plot_auto(node.outputs[0], label="input 0")
    plot_auto(node.outputs[1], label="input 1")
    plot_auto(node.outputs[2], label="input 2")
    close()

    savegraph(graph, f"output/{testname}.png")
