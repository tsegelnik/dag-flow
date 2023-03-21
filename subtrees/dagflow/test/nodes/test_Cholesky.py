#!/usr/bin/env python

from dagflow.exception import TypeFunctionError
from dagflow.graph import Graph
from dagflow.lib.Array import Array
from dagflow.lib.Cholesky import Cholesky
import numpy as np
import scipy
from pytest import raises
from dagflow.graphviz import savegraph

import pytest

@pytest.mark.parametrize("dtype", ('d', 'f'))
def test_Cholesky_00(dtype):
    inV = np.array([[10, 2,   1], [ 2, 12,  3], [ 1,  3, 13]], dtype=dtype)
    inV2 = inV@inV
    inD = np.diag(inV)
    inL2d1 = scipy.linalg.cholesky(inV, lower=True)
    inL2d2 = scipy.linalg.cholesky(inV2, lower=True)
    inL1d = np.sqrt(inD)

    with Graph(close=True) as graph:
        V1 = Array('V1', inV, mode='store')
        V2 = Array('V2', inV2, mode='store')
        D = Array('D', (inD), mode='store')
        chol2d = Cholesky('Cholesky 2d')
        chol1d = Cholesky('Cholesky 1d')
        (V1, V2) >> chol2d
        D >> chol1d

    assert V1.tainted==True
    assert V2.tainted==True
    assert chol2d.tainted==True
    assert chol1d.tainted==True

    result2d1 = chol2d.get_data(0)
    result2d2 = chol2d.get_data(1)
    result1d = chol1d.get_data(0)
    assert V1.tainted==False
    assert V2.tainted==False
    assert D.tainted==False
    assert chol2d.tainted==False
    assert chol1d.tainted==False

    assert np.allclose(inL2d1, result2d1, atol=0, rtol=0)
    assert np.allclose(inL2d2, result2d2, atol=0, rtol=0)
    assert np.allclose(inL1d, result1d, atol=0, rtol=0)

    savegraph(graph, f"output/test_Cholesky_00_{dtype}.png")

def test_Cholesky_01_typefunctions():
    inV = np.array([
        [10, 2,   1],
        [ 2, 12,  3],
        ], dtype='d')

    with Graph() as g1:
        V1 = Array('V1', inV, mode='store')
        chol1 = Cholesky('Cholesky')
        V1 >> chol1

    with Graph() as g2:
        V2 = Array('V2', inV[0], mode='store')
        chol2 = Cholesky('Cholesky')
        V2 >> chol1

    with raises(TypeFunctionError):
        g1.close()

    with raises(TypeFunctionError):
        g2.close()
