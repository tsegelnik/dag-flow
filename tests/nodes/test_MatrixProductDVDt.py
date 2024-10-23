from numpy import allclose, array, diag
from pytest import mark

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib.common import Array
from dagflow.lib.linear_algebra import MatrixProductDVDt


@mark.parametrize("dtype", ("d", "f"))
def test_MatrixProductDVDt_2d(dtype):
    left = array([[1, 2, 3], [3, 4, 5]], dtype=dtype)
    square = array(
        [
            [9, 2, 1],
            [0, 4, 2],
            [1.5, 3, 1],
        ],
        dtype=dtype,
    )

    with Graph(close_on_exit=True) as graph:
        l_array = Array("Left", left)
        s_array = Array("Square", square)

        prod = MatrixProductDVDt("MatrixProductDVDt2d")
        l_array >> prod.inputs["left"]
        s_array >> prod.inputs["square"]

    desired = left @ square @ left.T
    actual = prod.get_data("result")

    assert allclose(desired, actual, atol=0, rtol=0)

    savegraph(graph, f"output/test_MatrixProductDVDt_2d_{dtype}.png")


@mark.parametrize("dtype", ("d", "f"))
def test_MatrixProductDVDt_1d(dtype):
    left = array([[1, 2, 3], [3, 4, 5]], dtype=dtype)
    diagonal = array([9, 4, 5], dtype=dtype)

    with Graph(close_on_exit=True) as graph:
        l_array = Array("Left", left)
        s_array = Array("Diagonal", diagonal)

        prod = MatrixProductDVDt("MatrixProductDVDt1d")
        l_array >> prod.inputs["left"]
        s_array >> prod.inputs["square"]

    desired = left @ diag(diagonal) @ left.T
    actual = prod.get_data("result")

    assert allclose(desired, actual, atol=0, rtol=0)

    savegraph(graph, f"output/test_MatrixProductDVDt_1d_{dtype}.png")
