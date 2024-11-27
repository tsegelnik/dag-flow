from numpy import arange, sum
from pytest import mark

from dagflow.core.graph import Graph
from dagflow.plot.graphviz import savegraph
from dagflow.lib.common import Array
from dagflow.lib.summation import SumSq

debug = False


@mark.parametrize("dtype", ("d", "f"))
def test_SumSq_01(dtype):
    arrays_in = tuple(arange(12, dtype=dtype) * i for i in (1, 2, 3))
    arrays2_in = tuple(a**2 for a in arrays_in)

    with Graph(close_on_exit=True) as graph:
        arrays = tuple(Array("test", array_in) for array_in in arrays_in)
        sm = SumSq("sumsq")
        arrays >> sm

    output = sm.outputs[0]

    res = sum(arrays2_in, axis=0)

    assert sm.tainted == True
    assert all(output.data == res)
    assert sm.tainted == False

    arrays2_in = (arrays2_in[1],) + arrays2_in[1:]
    res = sum(arrays2_in, axis=0)
    assert arrays[0].set(arrays[1].get_data())
    assert sm.tainted == True
    assert all(output.data == res)
    assert sm.tainted == False

    savegraph(graph, f"output/test_SumSq_00_{dtype}.png")
