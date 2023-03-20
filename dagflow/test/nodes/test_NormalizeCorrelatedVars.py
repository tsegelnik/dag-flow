#!/usr/bin/env python

from numpy import arange
from dagflow.exception import TypeFunctionError

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib.Array import Array
from dagflow.lib.NormalizeCorrelatedVars import NormalizeCorrelatedVars
from dagflow.lib.Cholesky import Cholesky

from numpy import array, arange, allclose, sqrt
from scipy.linalg import solve_triangular, cholesky

import pytest
from pytest import raises

debug = False

@pytest.mark.parametrize('dtype', ('d', 'f'))
def test_NormalizeCorrelatedVars_00(dtype):
    inCentral = arange(3.0, dtype=dtype)*100.0
    inV = array([[10, 2,   1], [ 2, 12,  3], [ 1,  3, 13]], dtype=dtype)
    inD = inV.diagonal()
    inL = cholesky(inV, lower=True)
    inLd = sqrt(inD)
    inOffset = array((-10.0, 20.0, 30.0), dtype=dtype)
    inVec = inCentral + inOffset
    with Graph(close=True) as graph:
        matrix = Array('matrix', inV)
        diag = Array('diag', inD)
        Lmatrix = Cholesky('cholesky 1d')
        Ldiag = Cholesky('cholesky 2d')
        central = Array('central', inCentral)
        vec = Array('vec', inVec)
        norm1d_fwd = NormalizeCorrelatedVars('norm1d fwd')
        norm2d_fwd = NormalizeCorrelatedVars('norm2d fwd')

        norm1d_bwd = NormalizeCorrelatedVars('norm1d bwd', mode='backward')
        norm2d_bwd = NormalizeCorrelatedVars('norm2d bwd', mode='backward')

        central >> norm1d_fwd.inputs['central']
        central >> norm2d_fwd.inputs['central']
        central >> norm1d_bwd.inputs['central']
        central >> norm2d_bwd.inputs['central']

        matrix >> Lmatrix
        diag   >> Ldiag

        Lmatrix >> norm2d_fwd.inputs['matrix']
        Ldiag   >> norm1d_fwd.inputs['matrix']
        Lmatrix >> norm2d_bwd.inputs['matrix']
        Ldiag   >> norm1d_bwd.inputs['matrix']

        vec >> norm1d_fwd >> norm1d_bwd
        vec >> norm2d_fwd >> norm2d_bwd

    nodes = (
        matrix, diag,
        Lmatrix, Ldiag,
        central, vec,
        norm1d_fwd, norm2d_fwd,
        norm1d_bwd, norm2d_bwd,
    )

    assert all(node.tainted==True for node in nodes)
    back_matrix = norm2d_bwd.get_data(0)
    back_diag = norm1d_bwd.get_data(0)

    assert all(node.tainted==False for node in nodes)

    result_matrix = norm2d_fwd.get_data(0)
    result_diag = norm1d_fwd.get_data(0)

    norm1 = solve_triangular(inL, inOffset, lower=True)
    norm2 = inOffset/inLd

    if debug:
        print('V:', inV)
        print('Vdiag:', inD)
        print('L:', inL)
        print('Ldiag:', inLd)
        print('Central:', inCentral)
        print('In:', inVec)
        print('Offset:', inOffset)
        print('Norm 1:', norm1)
        print('Norm 2:', norm2)
        print('Rec 1:', back_matrix)
        print('Rec 2:', back_diag)
        print('Diff 1:', inVec-back_matrix)
        print('Diff 2:', inVec-back_diag)

    assert allclose(norm1, result_matrix, atol=0, rtol=0)
    assert allclose(norm2, result_diag, atol=0, rtol=0)
    assert allclose(inVec, back_matrix, atol=1.e-14, rtol=0)
    assert allclose(inVec, back_diag, atol=0, rtol=0)

    savegraph(graph, f"output/test_NormalizeCorrelatedVars_00_{dtype}.png")

def test_NormalizeCorrelatedVars_01(dtype='d'):
    inVec = arange(4.0, dtype=dtype)*100.0
    inV = array([[10, 2,   1], [ 2, 12,  3], [ 1,  3, 13]], dtype=dtype)
    inD = inV.diagonal()
    with Graph() as graph1:
        diag = Array('diag', inD)
        vec = Array('vec', inVec)
        norm1d_fwd = NormalizeCorrelatedVars('norm1d fwd')

        vec  >> norm1d_fwd.inputs['central']
        diag >> norm1d_fwd.inputs['matrix']

    with Graph() as graph2:
        matrix = Array('matrix', inV)
        vec = Array('vec', inVec)
        norm2d_fwd = NormalizeCorrelatedVars('norm2d fwd')

        vec >> norm2d_fwd.inputs['central']
        matrix >> norm2d_fwd.inputs['matrix']

    with raises(TypeFunctionError):
        graph1.close()

    with raises(TypeFunctionError):
        graph2.close()

