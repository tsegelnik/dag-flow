from numpy import arange
from pytest import mark

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib import Array
from dagflow.lib.LinearFunction import LinearFunction
from dagflow.makefcn import makefcn
from dagflow.parameters import Parameters
from dagflow.storage import NodeStorage


@mark.parametrize("pass_params", (False, True))
def test_makefcn_safe(testname, pass_params):
    n = 10
    x = arange(n, dtype="d")
    vals_in = [1.0, 2.0]
    vals_new = [3.0, 4.0]
    names = ("a", "b")

    with Graph(close=True) as graph:
        pars = Parameters.from_numbers(value=vals_in, names=names)
        storage = NodeStorage(
            {"parameters": {"all": {"a": pars._pars[0], "b": {"IDX1": pars._pars[1]}}}}
        )
        f = LinearFunction("ax+b")
        A, B = pars._pars
        A >> f("a")
        B >> f("b")
        Array("x", x) >> f

    res0 = f.outputs[0].data
    LF = makefcn(f, storage, safe=True, par_names=("a",) if pass_params else None)
    res1 = LF(a=vals_new[0], **{"parameters.all.b.IDX1": vals_new[1]})
    res2 = LF()

    # pars equal inital values
    assert A.value == vals_in[0]
    assert B.value == vals_in[1]
    # new result differs from the result of LF
    assert all(res1 == (vals_new[0] * x + vals_new[1]))
    assert all(res0 == (vals_in[0] * x + vals_in[1]))
    assert all(res2 == res0)

    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("pass_params", (False, True))
def test_makefcn_nonsafe(testname, pass_params):
    n = 10
    x = arange(n, dtype="d")
    vals_in = [1.0, 2.0]
    vals_new = [3.0, 4.0]
    names = ("a", "b.IDX1")

    with Graph(close=True) as graph:
        pars = Parameters.from_numbers(value=vals_in, names=names)
        storage = NodeStorage(
            {"parameters": {"all": {"a": pars._pars[0], "b": {"IDX1": pars._pars[1]}}}}
        )
        f = LinearFunction("ax+b")
        A, B = pars._pars
        A >> f("a")
        B >> f("b")
        Array("x", x) >> f

    res0 = f.outputs[0].data
    res0c = res0.copy()
    LF = makefcn(f, storage, safe=False, par_names=("a",) if pass_params else None)
    res1 = LF(a=vals_new[0], **{"parameters.all.b.IDX1": vals_new[1]})
    res2 = LF()

    # pars equal new values
    assert A.value == vals_new[0]
    assert B.value == vals_new[1]
    # new result is the same as the result of LF
    assert all(res0c == (vals_in[0] * x + vals_in[1]))
    assert all(res1 == (vals_new[0] * x + vals_new[1]))
    assert all(res1 == res0)
    assert all(res1 == res2)

    savegraph(graph, f"output/{testname}.png")
