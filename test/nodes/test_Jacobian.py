#!/usr/bin/env python

from typing import TYPE_CHECKING

from numpy import allclose, arange, eye, finfo, ones
from pytest import mark

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib import OneToOneNode
from dagflow.lib.Array import Array
from dagflow.lib.Concatenation import Concatenation
from dagflow.lib.Jacobian import Jacobian
from dagflow.parameters import GaussianParameter

if TYPE_CHECKING:
    from dagflow.input import Input


class LinearFunction(OneToOneNode):
    __slots__ = ("_a", "_b")
    _a: "Input"
    _b: "Input"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._a = self._add_input("a", positional=False)
        self._b = self._add_input("b", positional=False)

    def _typefunc(self) -> None:
        super()._typefunc()
        from dagflow.typefunctions import check_input_size

        check_input_size(self, ("a", "b"), exact=1)

    def _fcn(self):
        a = self._a.data[0]
        b = self._b.data[0]
        for inp, out in zip(self.inputs, self.outputs):
            for i in range(inp.dd.size):
                out.data[i] = a * inp.data[i] + b


@mark.parametrize("dtype", ("d", "f"))
def test_Jacobian_01(dtype, testname):
    """
    Test of the identity Jacobian.
    y_i = a_i, so the Jacobian is [dy/da_i]_i = 1 and [dy/da_i]_j = 0 when j != i
    """
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
    assert allclose(res, eye(size), atol=10 * finfo(dtype).resolution, rtol=0)

    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("dtype", ("d", "f"))
def test_Jacobian_02(dtype, testname):
    """
    Test of the linear function input.
    y = a x + b, so the Jacobian must be [dy/da, dy/db]_i = [x_i, 0]
    """
    size = 10
    scale = 0.5

    a = 2.3
    b = -3.2
    x = arange(size, dtype=dtype)
    sigmas = (0.5, 2.0)

    with Graph(close=True) as graph:
        X = Array("x", x)
        A = Array("a", [a])
        B = Array("b", [b])
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
        res, list(zip(x, ones(size, dtype=dtype))), atol=10 * finfo(dtype).resolution, rtol=0
    )

    savegraph(graph, f"output/{testname}.png")
