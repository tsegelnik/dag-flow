#!/usr/bin/env python

import numpy as np
from numpy import array, diag, sqrt, allclose, finfo
from dagflow.exception import TypeFunctionError
from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib import Array
from dagflow.lib import Cholesky
from pytest import mark, raises
from scipy import linalg


@mark.parametrize("dtype", ("d", "f"))
def test_Cholesky_00(testname, debug_graph, dtype):
    inV = array([[10, 2, 1], [2, 12, 3], [1, 3, 13]], dtype=dtype)
    inV2 = inV @ inV
    inD = diag(inV)
    inL2d1 = linalg.cholesky(inV, lower=True)
    inL2d2 = linalg.cholesky(inV2, lower=True)
    inL1d = sqrt(inD)

    with Graph(close=True, debug=debug_graph) as graph:
        V1 = Array("V1", inV, mode="store")
        V2 = Array("V2", inV2, mode="store")
        D = Array("D", (inD), mode="store")
        chol2d = Cholesky("Cholesky 2d")
        chol1d = Cholesky("Cholesky 1d")
        (V1, V2) >> chol2d
        D >> chol1d

    assert V1.tainted == True
    assert V2.tainted == True
    assert chol2d.tainted == True
    assert chol1d.tainted == True

    result2d1 = chol2d.get_data(0)
    result2d2 = chol2d.get_data(1)
    result1d = chol1d.get_data(0)
    assert V1.tainted == False
    assert V2.tainted == False
    assert D.tainted == False
    assert chol2d.tainted == False
    assert chol1d.tainted == False

    atol=finfo(dtype).resolution
    assert allclose(inL2d1, result2d1, atol=atol, rtol=0)
    assert allclose(inL2d2, result2d2, atol=atol, rtol=0)
    assert allclose(inL1d, result1d, atol=atol, rtol=0)

    savegraph(graph, f"output/{testname}.png")


def test_Cholesky_01_typefunctions():
    inV = array(
        [
            [10, 2, 1],
            [2, 12, 3],
        ],
        dtype="d",
    )

    with Graph() as g1:
        V1 = Array("V1", inV, mode="store")
        chol1 = Cholesky("Cholesky")
        V1 >> chol1

    with Graph() as g2:
        V2 = Array("V2", inV[0], mode="store")
        chol2 = Cholesky("Cholesky")
        V2 >> chol1

    with raises(TypeFunctionError):
        g1.close()

    with raises(TypeFunctionError):
        g2.close()
