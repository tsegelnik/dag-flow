#!/usr/bin/env python

from numpy import arange

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib.Array import Array
from dagflow.lib.CovmatrixFromCormatrix import CovmatrixFromCormatrix

from numpy import array, allclose, tril
import pytest

@pytest.mark.parametrize('dtype', ('d', 'f'))
def test_CovmatrixFromCormatrix_00(dtype):
    inSigma = arange(1.0, 4.0, dtype=dtype)
    inC = array([
        [1.0, 0.5, 0.0],
        [0.5, 1.0, 0.9],
        [0.0, 0.9, 1.0],
        ],
        dtype=dtype)
    with Graph(close=True) as graph:
        matrix = Array('matrix', inC)
        sigma = Array('sigma', inSigma)
        cov = CovmatrixFromCormatrix('covariance')

        sigma >> cov.inputs['sigma']
        matrix >> cov

    inV = inC * inSigma[:,None] * inSigma[None,:]
    V = cov.get_data()

    assert allclose(inV, V, atol=0, rtol=0)
    assert allclose(tril(V), tril(V.T), atol=0, rtol=0)

    savegraph(graph, f"output/test_CovmatrixFromCormatrix_00_{dtype}.png", show=['all'])

