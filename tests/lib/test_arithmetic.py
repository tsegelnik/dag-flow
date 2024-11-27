from numpy import allclose, arange, linspace, sqrt, square, sum
from pytest import mark

from dagflow.core.graph import Graph
from dagflow.plot.graphviz import savegraph
from dagflow.lib.arithmetic import Division, Product, Sqrt, Square, Sum
from dagflow.lib.common import Array


@mark.parametrize("dtype", ("d", "f"))
def test_Sum_01(testname, debug_graph, dtype):
    arrays_in = tuple(arange(12, dtype=dtype) * i for i in (1, 2, 3))

    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        arrays = tuple(Array(f"arr_{i}", array_in) for i, array_in in enumerate(arrays_in))
        sm = Sum("sum")
        arrays >> sm

    output = sm.outputs[0]

    res = sum(arrays_in, axis=0)

    assert sm.tainted == True
    assert all(output.data == res)
    assert sm.tainted == False

    for i in range(len(arrays_in)):
        arrays_in[i][:] = 2.3 * (i + 2) ** 2 + i
        res = arrays_in[0] + arrays_in[1] + arrays_in[2]
        arrays[i].outputs[0].set(2.3 * (i + 2) ** 2 + i)
        assert sm.tainted == True
        assert all(output.data == res)
        assert sm.tainted == False

    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("dtype", ("d", "f"))
def test_Product_01(testname, debug_graph, dtype):
    arrays_in = tuple(arange(12, dtype=dtype) * i for i in (1, 2, 3))

    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        arrays = tuple(Array(f"arr_{i}", array_in) for i, array_in in enumerate(arrays_in))
        prod = Product("prod")
        arrays >> prod

    output = prod.outputs[0]
    res = arrays_in[0] * arrays_in[1] * arrays_in[2]

    assert prod.tainted == True
    assert (output.data == res).all()
    assert prod.tainted == False

    for i in range(len(arrays_in)):
        arrays_in[i][:] = 2.3 * (i + 2) ** 2 + i
        res = arrays_in[0] * arrays_in[1] * arrays_in[2]
        arrays[i].outputs[0].set(2.3 * (i + 2) ** 2 + i)
        assert prod.tainted == True
        assert all(output.data == res)
        assert prod.tainted == False

    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("dtype", ("d", "f"))
def test_Division_01(testname, debug_graph, dtype):
    arrays_in = tuple(arange(12, dtype=dtype) * i + 1 for i in (1, 2, 3))

    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        arrays = tuple(Array(f"arr_{i}", array_in) for i, array_in in enumerate(arrays_in))
        div = Division("division")
        arrays >> div

    output = div.outputs[0]
    res = arrays_in[0] / arrays_in[1] / arrays_in[2]

    assert div.tainted == True
    assert (output.data == res).all()
    assert div.tainted == False

    for i in range(len(arrays_in)):
        arrays_in[i][:] = 2.3 * (i + 2) ** 2 + i
        res = arrays_in[0] / arrays_in[1] / arrays_in[2]
        arrays[i].outputs[0].set(2.3 * (i + 2) ** 2 + i)
        assert div.tainted == True
        assert all(output.data == res)
        assert div.tainted == False

    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("dtype", ("d", "f"))
@mark.parametrize("function", (square, sqrt))
def test_Powers_01(testname, debug_graph, function, dtype):
    if function == square:
        arrays_in = tuple(linspace(-10, 10, 101, dtype=dtype) * i for i in (1, 2, 3))
        cls = Square
        name = "Square"
    else:
        arrays_in = tuple(linspace(0, 10, 101, dtype=dtype) * i for i in (1, 2, 3))
        cls = Sqrt
        name = "Sqrt"

    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        arrays = tuple(
            Array(f"arr_{i}", array_in, label={"text": f"X axis {i}"})
            for i, array_in in enumerate(arrays_in)
        )
        node = cls(name)
        arrays >> node

    outputs = node.outputs
    ress = function(arrays_in)

    assert node.tainted == True
    assert all(output.dd.dtype == dtype for output in outputs)
    assert allclose(tuple(outputs.iter_data()), ress, rtol=0, atol=0)
    assert node.tainted == False

    for i in range(len(arrays_in)):
        arrays_in[i][:] = 2.3 * (i + 2) ** 2 + i
        ress = function(arrays_in)
        arrays[i].outputs[0].set(2.3 * (i + 2) ** 2 + i)
        assert node.tainted == True
        assert all(output.dd.dtype == dtype for output in outputs)
        assert allclose(tuple(outputs.iter_data()), ress, rtol=0, atol=0)
        assert node.tainted == False

    savegraph(graph, f"output/{testname}.png", show="full")
