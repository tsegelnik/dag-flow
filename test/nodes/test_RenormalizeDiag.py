#!/usr/bin/env python

from numpy import allclose, finfo, ones
from pytest import mark

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib.Array import Array
from dagflow.lib.RenormalizeDiag import RenormalizeDiag


@mark.parametrize("dtype", ("d", "f"))
@mark.parametrize("mode", ("diag", "offdiag"))
@mark.parametrize("ndiag", (1, 2, 3, 4))
def test_RenormalizeDiag(testname, ndiag, dtype, mode):
    size = 4
    scale = 2

    matrix = ones(shape=(size, size), dtype=dtype)
    with Graph(close=True) as graph:
        array_matrix = Array("matrix", matrix)
        scale_diag = Array("scale", [scale])

        prod = RenormalizeDiag("RenormalizeDiag", mode=mode, ndiag=ndiag)
        array_matrix >> prod
        scale_diag >> prod("scale")

    actual = prod.get_data()
    # NOTE: uncomment to print input and result
    print(f"matrix:\n{matrix}")
    print(f"result:\n{actual}")

    atol = finfo(dtype).resolution
    # check that sum of every row equals 1
    assert allclose(tuple(actual[i].sum() for i in range(size)), [1] * size, atol=atol, rtol=0)
    # TODO: check that we scaled correct diagonals

    ograph = f"output/{testname}.png"
    print(f"Write graph: {ograph}")
    savegraph(graph, ograph, show="all")
