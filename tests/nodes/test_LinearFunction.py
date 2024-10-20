from numpy import allclose, arange, array, finfo
from pytest import mark

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib.base import Array
from dagflow.lib.linear_algebra import LinearFunction


@mark.parametrize("dtype", ("d", "f"))
def test_LinearFunction_01(dtype, testname):
    size = 10

    a = 2.3
    b = -3.2
    x1 = arange(size, dtype=dtype)
    x2 = -x1

    with Graph(close_on_exit=True) as graph:
        X1 = Array("x1", x1)
        X2 = Array("x2", x2)
        A = Array("a", array([a], dtype=dtype))
        B = Array("b", array([b], dtype=dtype))
        Y = LinearFunction("f(x)", label="f(x)=ax+b")

        A >> Y("a")
        B >> Y("b")
        X1 >> Y
        X2 >> Y

    res1 = Y.outputs[0].data
    res2 = Y.outputs[1].data
    aa = A._output.data[0]  # get correct value with right dtype
    bb = B._output.data[0]  # get correct value with right dtype

    atol = finfo(dtype).resolution
    assert allclose(res1, x1 * aa + bb, atol=atol, rtol=0)
    assert allclose(res2, x2 * aa + bb, atol=atol, rtol=0)

    savegraph(graph, f"output/{testname}.png")
