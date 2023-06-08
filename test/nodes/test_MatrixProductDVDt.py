#!/usr/bin/env python

from dagflow.graph import Graph
from dagflow.lib.Array import Array
from dagflow.lib.MatrixProductDVDt import MatrixProductDVDt
from dagflow.graphviz import savegraph

import numpy as np
import pytest

@pytest.mark.parametrize("dtype", ('d', 'f'))
def test_MatrixProductDVDt_2d(dtype):
    left = np.array([[1, 2, 3], [3, 4, 5]], dtype=dtype)
    square = np.diag(np.array([9, 4, 5], dtype=dtype)) # construct a diagonal array

    with Graph(close=True) as graph:
        l_array = Array('Left', left)
        s_array = Array('Square', square)

        prod = MatrixProductDVDt("MatrixProductDVDt2d")
        l_array >> prod.inputs['left']
        s_array >> prod.inputs['square']

    desired = left @ square @ left.T
    actual = prod.get_data('result')

    assert np.allclose(desired, actual, atol=0, rtol=0)
    
    savegraph(graph, f"output/test_MatrixProductDVDt_2d_{dtype}.png")


@pytest.mark.parametrize("dtype", ('d', 'f'))
def test_MatrixProductDVDt_1d(dtype):
    left = np.array([[1, 2, 3], [3, 4, 5]], dtype=dtype)
    diagonal = np.array([9, 4, 5], dtype=dtype)

    with Graph(close=True) as graph:
        l_array = Array('Left', left)
        s_array = Array('Diagonal', diagonal)

        prod = MatrixProductDVDt("MatrixProductDVDt1d")
        l_array >> prod.inputs['left']
        s_array >> prod.inputs['square']

    desired = left @ np.diag(diagonal) @ left.T
    actual = prod.get_data('result')

    assert np.allclose(desired, actual, atol=0, rtol=0)

    savegraph(graph, f"output/test_MatrixProductDVDt_1d_{dtype}.png")


    
    