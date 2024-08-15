#!/usr/bin/env python
from numpy import allclose
from numpy import arange
from pytest import mark

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib.Array import Array
from dagflow.lib.NormalizeMatrix import NormalizeMatrix


@mark.parametrize("dtype", ("d", "f"))
@mark.parametrize("mode", ("rows", "columns"))
def test_NormalizeMatrix(dtype: str, mode: str):
    size = 4
    norm_columns = mode == "columns"

    matrix = (
        in_matrix := arange(1, size * (size + 1) + 1, dtype=dtype).reshape(
            size, size + 1
        )
    )
    if norm_columns:
        desired = matrix / matrix.sum(axis=0)
    else:
        desired = matrix / matrix.sum(axis=1)[:, None]

    with Graph(close_on_exit=True) as graph:
        array_matrix = Array("Matrix", matrix)

        prod = NormalizeMatrix("NormalizeMatrix", mode=mode)
        array_matrix >> prod

    actual = prod.get_data()
    assert allclose(desired, actual, atol=0, rtol=0)

    ograph = f"output/test_NormalizeMatrix_{dtype}_{mode}.png"
    print(f"Write graph: {ograph}")
    savegraph(graph, ograph, show="all")
