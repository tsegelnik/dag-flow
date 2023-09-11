#!/usr/bin/env python

from dagflow.graph import Graph
from dagflow.lib import Array
from dagflow.lib import VectorMatrixProduct
from dagflow.graphviz import savegraph

from numpy import arange, diag, allclose
from pytest import mark


@mark.parametrize("dtype", ("d", "f"))
@mark.parametrize("diag_matrix", (False, True))
@mark.parametrize("mode", ("row", "column"))
def test_VectorMatrixProduct(dtype: str, diag_matrix: bool, mode: str):
    size = 4
    is_column = mode == "column"

    matrix = (
        in_matrix := arange(1, size * (size + 1) + 1, dtype=dtype).reshape(
            size, size + 1
        )
    )

    if diag_matrix:
        matrix = diag(in_matrix[:size, :size])
        in_matrix = diag(matrix)

    if is_column:
        vector = arange(1, matrix.shape[-1] + 1, dtype=dtype)
        column = vector[:, None]
        desired = (in_matrix @ column).ravel()
    else:
        vector = arange(1, matrix.shape[0] + 1, dtype=dtype)
        row = vector[None, :]
        desired = (row @ in_matrix).ravel()

    with Graph(close=True) as graph:
        array_vector = Array("Vector", vector)
        array_matrix = Array("Matrix", matrix)

        prod = VectorMatrixProduct("VectorMatrixProduct", mode=mode)
        array_vector >> prod.inputs["vector"]
        array_matrix >> prod.inputs["matrix"]

    actual = prod.get_data()
    assert allclose(desired, actual, atol=0, rtol=0)
    assert len(actual.shape) == 1

    smatrix = diag_matrix and "diag" or "block"
    ograph = f"output/test_VectorMatrixProduct_{dtype}_{smatrix}_{mode}.png"
    print(f"Write graph: {ograph}")
    savegraph(graph, ograph)
