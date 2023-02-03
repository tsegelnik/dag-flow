#!/usr/bin/env python

from dagflow.exception import TypeFunctionError
from dagflow.graph import Graph
from dagflow.lib.Array import Array
from dagflow.lib.Cholesky import Cholesky
import numpy as np
from pytest import raises

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
