#!/usr/bin/env python

from dagflow.graph import Graph
from dagflow.lib.Array import Array
from dagflow.lib.Cholesky import Cholesky
import numpy as np

def test_Cholesky_00():
    inV = np.array([
        [10, 2,   1],
        [ 2, 12,  3],
        [ 1,  3, 13],
        ], dtype='d')
    inL = np.linalg.cholesky(inV)

    with Graph(close=True):
        V = Array('V', inV, mode='store')
        chol = Cholesky('Cholesky')
        V >> chol

    assert V.tainted==True
    assert chol.tainted==True

    result = chol.get_data(0)
    assert V.tainted==False
    assert chol.tainted==False

    assert np.allclose(inL, result, atol=0, rtol=0)

