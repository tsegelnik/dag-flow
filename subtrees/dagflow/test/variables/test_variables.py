#!/usr/bin/env python

from dagflow.lib import Array
from dagflow.variable import GaussianParameters
from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.exception import CriticalError

from numpy import square, allclose
import pytest

@pytest.mark.parametrize('mode', ('single', 'uncorr', 'cov', 'cov1d'))
def test_variables_00_variable(mode) -> None:
    value_in    = [1.1, 1.8, 5.0]
    central_in  = [1.0, 2.0, 3.0]
    sigma_in    = [1.0, 0.5, 2.0]
    corrs_in    = [-0.1, 0.5, -0.9] # 01, 02, 12
    variance_in = square(sigma_in)
    zeros_in    = [0.0, 0.0, 0.0]

    if mode=='single':
        value_in = value_in[:1]
        central_in = central_in[:1]
        sigma_in = sigma_in[:1]
        zeros_in = zeros_in[:1]

    with Graph(debug=False, close=False) as graph:
        value   = Array("variable", value_in, mode='store_weak', mark='v')
        central = Array("central",  central_in, mark='v₀')

        if mode in ('single', 'uncorr', 'cor'):
            sigma = Array("sigma", sigma_in, mark='σ')

        if mode in ('single', 'uncorr'):
            gp = GaussianParameters(value, central, sigma=sigma)
        elif mode=='cov':
            covariance = Array("covariance", [
                    [variance_in[0],                      corrs_in[0]*sigma_in[0]*sigma_in[1], corrs_in[1]*sigma_in[0]*sigma_in[2]],
                    [corrs_in[0]*sigma_in[0]*sigma_in[1], variance_in[1],                      corrs_in[2]*sigma_in[1]*sigma_in[2]],
                    [corrs_in[1]*sigma_in[0]*sigma_in[2], corrs_in[2]*sigma_in[1]*sigma_in[2], variance_in[2]]
                                ],
                               mark='V')
            gp = GaussianParameters(value, central, covariance=covariance)
        elif mode=='cov1d':
            covariance = Array("covariance", variance_in, mark='diag(V)')
            gp = GaussianParameters(value, central, covariance=covariance)
        elif mode=='cor':
            correlation = Array("correlation", [
                [1.0,         corrs_in[0], corrs_in[1]],
                [corrs_in[0], 1.0,         corrs_in[2]],
                [corrs_in[1], corrs_in[2], 1.0],
                ], mark='C')
            gp = GaussianParameters(value, central, sigma=sigma, correlation=correlation)
        else:
            raise RuntimeError(f"Invalid mode {mode}")

    try:
        graph.close()
    except CriticalError as error:
        savegraph(graph, f"output/test_variables_00_{mode}.png")
        raise error

    value_out0 = gp.value.data.copy()
    normvalue_out0 = gp.normvalue.data
    assert allclose(value_in, value_out0, atol=0, rtol=0)
    assert all(normvalue_out0!=0)

    gp.normvalue.set(zeros_in)
    value_out1 = gp.value.data
    normvalue_out1 = gp.normvalue.data
    assert allclose(central_in, value_out1, atol=0, rtol=0)
    assert allclose(normvalue_out1, 0.0, atol=0, rtol=0)

    gp.value.set(value_out0)
    value_out2 = gp.value.data
    normvalue_out2 = gp.normvalue.data
    assert allclose(value_in, value_out2, atol=0, rtol=0)
    assert allclose(normvalue_out2, normvalue_out0, atol=0, rtol=0)

    savegraph(graph, f"output/test_variables_00_{mode}.png", show=['all'])
    savegraph(graph, f"output/test_variables_00_{mode}.pdf", show=['all'])

