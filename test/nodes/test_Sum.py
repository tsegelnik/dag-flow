#!/usr/bin/env python

from dagflow.graph import Graph
from dagflow.lib.Array import Array
from dagflow.lib.Sum import Sum
from dagflow.graphviz import savegraph

from numpy import arange, sum
import pytest

debug = False

@pytest.mark.parametrize('dtype', ('d', 'f'))
def test_Sum_01(dtype):
    arrays_in = tuple(arange(12, dtype=dtype)*i for i in (1, 2, 3))

    with Graph(close=True) as graph:
        arrays = tuple(Array('test', array_in) for array_in in arrays_in)
        sm = Sum('sum')
        arrays >> sm

    output = sm.outputs[0]

    res = sum(arrays_in, axis=0)

    assert sm.tainted==True
    assert all(output.data==res)
    assert sm.tainted==False

    arrays_in = (arrays_in[1],) + arrays_in[1:]
    res = sum(arrays_in, axis=0)
    assert arrays[0].set(arrays[1].get_data())
    assert sm.tainted==True
    assert all(output.data==res)
    assert sm.tainted==False

    savegraph(graph, f"output/test_sum_00_{dtype}.png")
