from numpy import allclose, array
from pytest import mark

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib.common import Array
from dagflow.lib.linear_algebra import MatrixProductDDt


@mark.parametrize("dtype", ("d", "f"))
def test_MatrixProductDVDt_2d(dtype):
    left = array([[1, 2, 3], [3, 4, 5]], dtype=dtype)

    with Graph(close_on_exit=True) as graph:
        l_array = Array("Left", left)

        prod = MatrixProductDDt("MatrixProductDDt2d")
        l_array >> prod

    desired = left @ left.T
    actual = prod.get_data("result")

    assert allclose(desired, actual, atol=0, rtol=0)

    savegraph(graph, f"output/test_MatrixProductDDt_2d_{dtype}.png")
