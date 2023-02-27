#!/usr/bin/env python

from dagflow.graph import Graph
from dagflow.lib.Array import Array
from dagflow.lib.SumMatOrDiag import SumMatOrDiag
from dagflow.graphviz import savegraph

from numpy import arange, diag, allclose
import pytest

debug = False

@pytest.mark.parametrize('dtype', ('d', 'f'))
def test_SumMatOrDiag_01(dtype):
    for size in (5, 4):
        array1  = arange(size, dtype=dtype)+1.0
        array2  = arange(size, dtype=dtype)*3
        matrix1 = arange(size*size, dtype=dtype).reshape(size, size)+1.0 
        matrix2 = arange(size*size, dtype=dtype).reshape(size, size)*2.5
        arrays_in = (array1, array2, matrix1, matrix2)

        combinations = ((0,), (2,), (0, 1), (0, 2), (2, 0), (0, 1, 2), (2, 3), (0, 1, 2, 3))

        sms = []

        with Graph(close=True) as graph:
            arrays = tuple(Array(f'test {i}', array_in) for i, array_in in enumerate(arrays_in))

            for cmb in combinations:
                sm = SumMatOrDiag(f'sum {cmb}')
                tuple(arrays[i] for i in cmb) >> sm
                sms.append(sm)

        for cmb, sm in zip(combinations, sms):
            res = 0.0
            all1d = True
            for i in cmb:
                array_in = arrays_in[i]
                if len(array_in.shape)==1:
                    array_in = diag(array_in)
                else:
                    all1d = False
                res += array_in

            if all1d:
                res = diag(res)

            assert sm.tainted==True
            output = sm.outputs[0]
            assert allclose(output.data, res, rtol=0, atol=0)
            assert sm.tainted==False

        savegraph(graph, f"output/test_SumMatOrDiag_00_{dtype}_{size}.png", show='all')
