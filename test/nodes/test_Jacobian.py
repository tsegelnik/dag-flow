#!/usr/bin/env python

from numpy import allclose, arange, eye, finfo, ones
from pytest import mark

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib.Array import Array
from dagflow.lib.Concatenation import Concatenation
from dagflow.lib.Jacobian import Jacobian
from dagflow.parameters import GaussianParameter


@mark.parametrize("dtype", ("d", "f"))
def test_Jacobian_01(dtype, testname):
    size = 10
    scale = 0.5
    values = arange(size, dtype=dtype)
    sigmas = ones(size, dtype=dtype)

    with Graph(close=True) as graph:
        valueslist = [Array(f"arr_{i:02d}", [val]) for i, val in enumerate(values)]
        sigmaslist = [Array(f"sigma_{i:02d}", [sigma]) for i, sigma in enumerate(sigmas)]
        pars = [
            GaussianParameter(
                parent=None,
                value_output=value._output,
                central_output=value._output,
                normvalue_output=value._output,
                sigma_output=sigma._output,
            )
            for value, sigma in zip(valueslist, sigmaslist)
        ]
        jac = Jacobian(
            "Jacobian",
            scale=scale,
            parameters=pars,
        )
        parsconcat = Concatenation("parameters")
        valueslist >> parsconcat
        parsconcat >> jac

    res = jac.outputs[0].data
    print(res)
    assert allclose(res, eye(size), atol=10 * finfo(dtype).resolution, rtol=0)

    savegraph(graph, f"output/{testname}.png")
