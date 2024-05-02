from numpy import arange

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib import Array
from dagflow.lib.LinearFunction import LinearFunction
from dagflow.makefcn import makefcn
from dagflow.parameters import Parameters
from dagflow.storage import NodeStorage


def test_makefcn_safe(testname):
    n = 10
    x = arange(n, dtype="d")
    vals_in = [1.0, 2.0]
    vals_new = [3.0, 4.0]
    names = ("a", "b")

    with Graph(close=True) as graph:
        pars = Parameters.from_numbers(value=vals_in, names=names)
        storage = NodeStorage({"parameter": {"all": dict(zip(names, pars._pars))}})
        f = LinearFunction("ax+b")
        A, B = pars._pars
        A >> f("a")
        B >> f("b")
        Array("x", x) >> f

    res0 = f.outputs[0].data
    LF = makefcn(f, storage, safe=True)
    res1 = LF(a=vals_new[0], b=vals_new[1])

    # pars equal inital values
    assert A.value == vals_in[0]
    assert B.value == vals_in[1]
    # new result differs from the result of LF
    assert all(res1 == vals_new[0] * x + vals_new[1])
    assert all(res0 == vals_in[0] * x + vals_in[1])

    savegraph(graph, f"output/{testname}.png")


def test_makefcn_nonsafe(testname):
    n = 10
    x = arange(n, dtype="d")
    vals_in = [1.0, 2.0]
    vals_new = [3.0, 4.0]
    names = ("a", "b")

    with Graph(close=True) as graph:
        pars = Parameters.from_numbers(value=vals_in, names=names)
        storage = NodeStorage({"parameter": {"all": dict(zip(names, pars._pars))}})
        f = LinearFunction("ax+b")
        A, B = pars._pars
        A >> f("a")
        B >> f("b")
        Array("x", x) >> f

    res0 = f.outputs[0].data
    res0c = res0.copy()
    LF = makefcn(f, storage, safe=False)
    res1 = LF(a=vals_new[0], b=vals_new[1])

    # pars equal new values
    assert A.value == vals_new[0]
    assert B.value == vals_new[1]
    # new result is the same as the result of LF
    assert all(res0c == vals_in[0] * x + vals_in[1])
    assert all(res1 == vals_new[0] * x + vals_new[1])
    assert all(res1 == res0)

    savegraph(graph, f"output/{testname}.png")
