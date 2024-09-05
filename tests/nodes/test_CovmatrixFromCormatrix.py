from numpy import allclose
from numpy import arange
from numpy import array
from numpy import tril
from pytest import mark

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib import Array
from dagflow.lib import CovmatrixFromCormatrix


@mark.parametrize("dtype", ("d", "f"))
def test_CovmatrixFromCormatrix_00(testname, debug_graph, dtype):
    inSigma = arange(1.0, 4.0, dtype=dtype)
    inC = array(
        [
            [1.0, 0.5, 0.0],
            [0.5, 1.0, 0.9],
            [0.0, 0.9, 1.0],
        ],
        dtype=dtype,
    )
    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        matrix = Array("matrix", inC)
        sigma = Array("sigma", inSigma)
        cov = CovmatrixFromCormatrix("covariance")

        sigma >> cov.inputs["sigma"]
        matrix >> cov

    inV = inC * inSigma[:, None] * inSigma[None, :]
    V = cov.get_data()

    assert allclose(inV, V, atol=0, rtol=0)
    assert allclose(tril(V), tril(V.T), atol=0, rtol=0)

    savegraph(graph, f"output/{testname}.png", show=["all"])
