#!/usr/bin/env python

from numpy import allclose, arange, finfo
from pytest import mark

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib.Array import Array
from dagflow.lib.RenormalizeDiag import RenormalizeDiag


@mark.parametrize("dtype", ("d", "f"))
@mark.parametrize("mode", ("diag", "offdiag"))
@mark.parametrize("ndiag", (1, 2, 3, 4))
@mark.parametrize("scale", (2.0, 0.33))
def test_RenormalizeDiag(testname, ndiag, dtype, mode, scale):
    size = 4
    matrix = arange(size**2, dtype=dtype).reshape(size, size)
    with Graph(close=True) as graph:
        array_matrix = Array("matrix", matrix)
        scale_diag = Array("scale", [scale])

        prod = RenormalizeDiag("RenormalizeDiag", mode=mode, ndiag=ndiag)
        array_matrix >> prod
        scale_diag >> prod("scale")

    actual = prod.get_data()
    # NOTE: uncomment to print input and result
    # print(f"matrix:\n{matrix}")
    # print(f"result:\n{actual}")

    atol = finfo(dtype).resolution
    # check that sum of every column equals 1
    assert allclose(
        tuple(actual[:, icol].sum() for icol in range(size)), [1] * size, atol=atol, rtol=0
    )

    # check that we scaled correct diagonals
    desired = matrix.copy()
    idiag = 1
    if mode == "diag":
        for i in range(size):
            desired[i, i] *= scale
        while idiag < ndiag:
            i = 0
            while i + idiag < size:
                desired[i + idiag, i] *= scale
                desired[i, i + idiag] *= scale
                i += 1
            idiag += 1
    else:
        desired *= scale
        for i in range(size):
            desired[i, i] = matrix[i, i]
        while idiag < ndiag:
            i = 0
            while i + idiag < size:
                desired[i + idiag, i] = matrix[i + idiag, i]
                desired[i, i + idiag] = matrix[i, i + idiag]
                i += 1
            idiag += 1
    for icol in range(size):
        desired[:, icol] /= ssum if (ssum := desired[:, icol].sum()) != 0.0 else 1

    # NOTE: uncomment to print desired
    # print(f"desired:\n{desired}")
    assert allclose(desired, actual, atol=atol, rtol=0)

    ograph = f"output/{testname}.png"
    print(f"Write graph: {ograph}")
    savegraph(graph, ograph, show="all")
