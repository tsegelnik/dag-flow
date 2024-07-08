#!/usr/bin/env python

from numpy import allclose, arange, array, eye, finfo, ones
from pytest import mark

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib.Array import Array
from dagflow.lib.Concatenation import Concatenation
from dagflow.lib.Jacobian import Jacobian
from dagflow.lib.LinearFunction import LinearFunction
from dagflow.parameters import GaussianParameter


@mark.parametrize("dtype", ("d", "f"))
def test_Jacobian_01(dtype, testname):
    """
    Test of the identity Jacobian.
    y_i = a_i, so the Jacobian is [dy/da_i]_i = 1 and [dy/da_i]_j = 0 when j != i
    """
    size = 10
    scale = 1
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
    assert allclose(res, eye(size), atol=2 * finfo(dtype).resolution, rtol=0)

    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("dtype", ("d", "f"))
def test_Jacobian_02(dtype, testname):
    """
    Test of the linear function input.
    y = a x + b, so the Jacobian must be [dy/da, dy/db]_i = [x_i, 1]
    """
    size = 10
    scale = 1

    a = 2.3
    b = -3.2
    x = arange(size, dtype=dtype)
    sigmas = (0.5, 2.0)

    with Graph(close=True) as graph:
        X = Array("x", x)
        A = Array("a", array([a], dtype=dtype))
        B = Array("b", array([b], dtype=dtype))
        Y = LinearFunction("f(x)", label="f(x)=ax+b")
        A >> Y("a")
        B >> Y("b")
        X >> Y

        sigmaslist = [Array(f"sigma_{i:02d}", [sigma]) for i, sigma in enumerate(sigmas)]
        pars = [
            GaussianParameter(
                parent=None,
                value_output=value._output,
                central_output=value._output,
                normvalue_output=value._output,
                sigma_output=sigma._output,
            )
            for value, sigma in zip((A, B), sigmaslist)
        ]
        jac = Jacobian(
            "Jacobian",
            scale=scale,
            parameters=pars,
        )
        Y >> jac

    res = jac.outputs[0].data
    assert allclose(
        res, list(zip(x, ones(size, dtype=dtype))), atol=5 * finfo(dtype).resolution, rtol=0
    )

    savegraph(graph, f"output/{testname}.png")
