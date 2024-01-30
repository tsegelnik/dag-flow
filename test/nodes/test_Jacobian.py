#!/usr/bin/env python

from numpy import allclose, arange, eye, finfo
from pytest import mark

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib.Jacobian import Jacobian
from dagflow.parameters import Parameters


@mark.parametrize("dtype", ("d", "f"))
@mark.parametrize("parameters_mode", ("list", "Parameters"))
def test_Jacobian(dtype, parameters_mode, testname):
    size = 10
    reldelta = 0.1
    step = 0.1
    values = arange(size, dtype=dtype)
    names = tuple(f"par_{i:02d}" for i in range(size))

    with Graph(close=True) as graph:
        pars = Parameters.from_numbers(value=values, names=names, dtype=dtype)
        jac = Jacobian(
            "Jacobian",
            reldelta=reldelta,
            step=step,
            parameters=pars if parameters_mode == "Parameters" else pars._pars,
        )
        pars._value_node.outputs[0] >> jac("func")

    res = jac._jacobian.data
    assert allclose(res, eye(size), atol=finfo(dtype).resolution, rtol=0)

    savegraph(graph, f"output/{testname}.png")
