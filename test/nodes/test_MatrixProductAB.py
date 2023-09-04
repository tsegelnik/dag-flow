#!/usr/bin/env python

from dagflow.graph import Graph
from dagflow.lib.Array import Array
from dagflow.lib.MatrixProductAB import MatrixProductAB
from dagflow.graphviz import savegraph

from numpy import arange, diag, allclose
from pytest import mark


@mark.parametrize("dtype", ("d", "f"))
@mark.parametrize("diag_left", (False, True))
@mark.parametrize("diag_right", (False, True))
def test_MatrixProductAB(dtype: str, diag_left: bool, diag_right: bool):
    size = 3
    left = (in_left := arange(1, size*(size+1)+1, dtype=dtype).reshape(size+1,size))
    right = (in_right := arange(1, size*(size+1)+1, dtype=dtype).reshape(size,size+1))

    if diag_left:
        left = diag(in_left[:size,:size])
        in_left = diag(left)
    if diag_right:
        right = diag(in_right[:size,:size])
        in_right = diag(right)

    with Graph(close=True) as graph:
        array_left = Array("Left", left)
        array_right = Array("Right", right)

        prod = MatrixProductAB("MatrixProductAB")
        array_left >> prod.inputs["left"]
        array_right >> prod.inputs["right"]

    desired = in_left @ in_right
    if diag_left and diag_right:
        desired = diag(desired)

    actual = prod.get_data()
    assert allclose(desired, actual, atol=0, rtol=0)
    assert (diag_left and diag_right)==(len(actual.shape)==1)

    sleft = diag_left and 'diag' or 'block'
    sright = diag_right and 'diag' or 'block'
    ograph = f"output/test_MatrixProductAB_{dtype}_{sleft}_{sright}.png"
    print(f'Write graph: {ograph}')
    savegraph(graph, ograph)

