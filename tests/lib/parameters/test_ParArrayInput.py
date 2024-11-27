from numpy import allclose, arange, ones
from pytest import mark

from dagflow.core.graph import Graph
from dagflow.plot.graphviz import savegraph
from dagflow.lib.common import Array
from dagflow.lib.parameters import ParArrayInput
from dagflow.parameters import Parameters


@mark.parametrize("dtype", ("d", "f"))
@mark.parametrize("parameters_mode", ("list", "Parameters"))
def test_ParArrayInput(dtype, parameters_mode, testname):
    size = 10
    values_initial = ones(size, dtype=dtype)
    values_new = arange(size, dtype=dtype) + 2
    names = tuple(f"par_{i:02d}" for i in range(size))

    with Graph(close_on_exit=True) as graph:
        pars = Parameters.from_numbers(value=values_initial, names=names, dtype=dtype)
        arr = Array("new values", values_new)
        parinp = ParArrayInput(
            "ParArrayInput", parameters=pars if parameters_mode == "Parameters" else pars._pars
        )
        arr >> parinp

    parinp.touch()
    res = tuple(par.value for par in pars._pars)
    assert allclose(res, values_new, atol=0, rtol=0)

    out = parinp._values.parent_output
    out.set(values_initial)
    parinp.touch()
    res = tuple(par.value for par in pars._pars)
    assert allclose(res, values_initial, atol=0, rtol=0)

    savegraph(graph, f"output/{testname}.png")
