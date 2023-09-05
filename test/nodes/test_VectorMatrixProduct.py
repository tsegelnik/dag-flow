#!/usr/bin/env python

from dagflow.graph import Graph
from dagflow.lib.Array import Array
from dagflow.lib.VectorMatrixProduct import VectorMatrixProduct
from dagflow.graphviz import savegraph

from numpy import arange, diag, allclose
from pytest import mark


@mark.parametrize("dtype", ("d", "f"))
@mark.parametrize("diag_matrix", (False, True))
def test_VectorMatrixProduct(dtype: str, diag_matrix: bool):
    size = 4
    vector = arange(1, size+1, dtype=dtype)
    matrix = (in_matrix := arange(1, size*(size+1)+1, dtype=dtype).reshape(size,size+1))

    if diag_matrix:
        matrix = diag(in_matrix[:size,:size])
        in_matrix = diag(matrix)

    with Graph(close=True) as graph:
        array_vector = Array("Vector", vector)
        array_matrix = Array("Matrix", matrix)

        prod = VectorMatrixProduct("VectorMatrixProduct")
        array_vector >> prod.inputs["vector"]
        array_matrix >> prod.inputs["matrix"]

    desired = vector @ in_matrix

    actual = prod.get_data()
    assert allclose(desired, actual, atol=0, rtol=0)
    assert diag_matrix==(len(actual.shape)==1)

    smatrix = diag_matrix and 'diag' or 'block'
    ograph = f"output/test_VectorMatrixProduct_{dtype}_{smatrix}.png"
    print(f'Write graph: {ograph}')
    savegraph(graph, ograph)

