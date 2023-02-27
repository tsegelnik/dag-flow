#!/usr/bin/env python

from numpy import arange
from dagflow.exception import TypeFunctionError

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib.Array import Array
from dagflow.lib.NormalizeCorrelatedVars2 import NormalizeCorrelatedVars2
from dagflow.lib.Cholesky import Cholesky

from numpy import array, arange, allclose, sqrt, full_like, zeros_like, ones_like
from scipy.linalg import solve_triangular, cholesky

import pytest
from pytest import raises

debug = False

@pytest.mark.parametrize('dtype', ('d', 'f'))
def test_NormalizeCorrelatedVars2_00(dtype):
    inCentral = arange(3.0, dtype=dtype)*100.0
    inV = array([[10, 2,   1], [ 2, 12,  3], [ 1,  3, 13]], dtype=dtype)
    inD = inV.diagonal()
    inL = cholesky(inV, lower=True)
    inLd = sqrt(inD)
    inOffset = array((-10.0, 20.0, 30.0), dtype=dtype)
    inVec = inCentral + inOffset
    inNorm = full_like(inVec, -100)
    with Graph(close=True) as graph:
        matrix = Array('matrix', inV)
        diag = Array('diag', inD)
        Lmatrix = Cholesky('cholesky 1d')
        Ldiag = Cholesky('cholesky 2d')
        central = Array('central', inCentral)
        value1d = Array('vec 1d', inVec, mode='store_weak')
        normvalue1d = Array('normvalue 1d', inNorm, mode='store_weak')
        value2d = Array('vec 2d', inVec, mode='store_weak')
        normvalue2d = Array('normvalue 2d', inNorm, mode='store_weak')
        norm1d = NormalizeCorrelatedVars2('norm1d')
        norm2d = NormalizeCorrelatedVars2('norm2d')

        central >> norm1d.inputs['central']
        central >> norm2d.inputs['central']

        matrix >> Lmatrix
        diag   >> Ldiag

        Lmatrix >> norm2d.inputs['matrix']
        Ldiag   >> norm1d.inputs['matrix']

        (value1d, normvalue1d) >> norm1d
        (value2d, normvalue2d) >> norm2d


    nodes = (
        matrix, diag,
        Lmatrix, Ldiag,
        central,
        value1d, normvalue1d,
        value1d, normvalue2d,
        norm1d, norm2d,
    )

    assert all(node.tainted==True for node in nodes)
    norm_matrix = norm2d.get_data(1)
    norm_diag = norm1d.get_data(1)
    back_matrix = norm2d.get_data(0)
    back_diag = norm1d.get_data(0)

    assert all(node.tainted==False for node in nodes)
    assert all(inNorm!=back_matrix)
    assert all(inNorm!=back_diag)

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

    assert allclose(norm1, norm_matrix, atol=0, rtol=0)
    assert allclose(norm2, norm_diag, atol=0, rtol=0)
    assert allclose(inVec, back_matrix, atol=1.e-14, rtol=0)
    assert allclose(inVec, back_diag, atol=0, rtol=0)

    inZeros = zeros_like(inVec)
    normvalue1d.set(inZeros)
    normvalue2d.set(inZeros)
    back_matrix = norm2d.get_data(0)
    back_diag = norm1d.get_data(0)
    norm_matrix = norm2d.get_data(1)
    norm_diag = norm1d.get_data(1)
    assert allclose(inZeros, norm_matrix, atol=0, rtol=0)
    assert allclose(inZeros, norm_diag, atol=0, rtol=0)
    assert allclose(inCentral, back_matrix, atol=1.e-14, rtol=0)
    assert allclose(inCentral, back_diag, atol=0, rtol=0)

    inOnes = ones_like(inVec)
    normvalue1d.set(inOnes)
    normvalue2d.set(inOnes)
    back_matrix = norm2d.get_data(0)
    back_diag = norm1d.get_data(0)
    norm_matrix = norm2d.get_data(1)
    norm_diag = norm1d.get_data(1)
    checkDiag = inCentral + inLd
    checkMatrix = inCentral + inL@inOnes
    assert allclose(inOnes, norm_matrix, atol=0, rtol=0)
    assert allclose(inOnes, norm_diag, atol=0, rtol=0)
    assert allclose(checkMatrix, back_matrix, atol=1.e-14, rtol=0)
    assert allclose(checkDiag, back_diag, atol=0, rtol=0)

    norm2d._immediate = True
    norm1d._immediate = True
    value1d.set(inCentral)
    value2d.set(inCentral)
    norm_matrix = norm2d.outputs[1]._data
    norm_diag = norm1d.outputs[1]._data
    back_matrix = norm2d.outputs[0]._data
    back_diag = norm1d.outputs[0]._data
    assert allclose(inZeros, norm_matrix, atol=0, rtol=0)
    assert allclose(inZeros, norm_diag, atol=0, rtol=0)
    assert allclose(inCentral, back_matrix, atol=1.e-14, rtol=0)
    assert allclose(inCentral, back_diag, atol=0, rtol=0)


    savegraph(graph, f"output/test_NormalizeCorrelatedVars2_00_{dtype}.png", show=['all'])

def test_NormalizeCorrelatedVars2_01(dtype='d'):
    inVec = arange(4.0, dtype=dtype)*100.0
    inNorm = full_like(inVec, -100)
    inV = array([[10, 2,   1], [ 2, 12,  3], [ 1,  3, 13]], dtype=dtype)
    inD = inV.diagonal()
    with Graph() as graph1:
        diag = Array('diag', inD)
        vec = Array('vec', inVec, mode='store_weak')
        nvec = Array('vec', inNorm, mode='store_weak')
        norm1d = NormalizeCorrelatedVars2('norm1d')

        vec  >> norm1d.inputs['central']
        diag >> norm1d.inputs['matrix']

        (vec, nvec) >> norm1d

    with Graph() as graph2:
        matrix = Array('matrix', inV)
        vec = Array('vec', inVec, mode='store_weak')
        nvec = Array('vec', inNorm, mode='store_weak')
        norm2d = NormalizeCorrelatedVars2('norm2d')

        vec >> norm2d.inputs['central']
        matrix >> norm2d.inputs['matrix']

        (vec, nvec) >> norm2d

    with raises(TypeFunctionError):
        graph1.close()

    with raises(TypeFunctionError):
        graph2.close()

