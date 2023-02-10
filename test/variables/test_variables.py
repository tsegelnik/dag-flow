#!/usr/bin/env python

from dagflow.lib import Array
from dagflow.lib.NormalizeCorrelatedVars import NormalizeCorrelatedVars
from dagflow.variable import Parameters, GaussianParameters
from dagflow.graph import Graph
from dagflow.graphviz import savegraph

from numpy import square
import pytest

# def test_variables_00_parameter() -> None:
#     pass


@pytest.mark.parametrize('mode', ('single', 'uncorr', 'cov', 'cov1d'))
def test_variables_00_variable(mode) -> None:
    value_in = [1.1, 1.8, 5.0]
    central_in = [1.0, 2.0, 3.0]
    sigma_in = [1.0, 0.5, 2.0]
    var_in = square(sigma_in)

    with Graph(close=True) as graph:
        if mode=='single':
            value   = Array("variable", value_in[:1],   mode='store_weak', mark='v')
            central = Array("central",  central_in[:1], mode='store_weak', mark='v₀')
        else:
            value   = Array("variable", value_in,       mode='store_weak', mark='v')
            central = Array("central",  central_in,     mode='store_weak', mark='v₀')

        if mode=='single':
            sigma   = Array("central",  [1.0], mode='store_weak', mark='σ')
            gp = GaussianParameters(value, central, sigma=sigma)
        elif mode=='uncorr':
            sigma   = Array("sigma",    sigma_in, mode='store_weak', mark='σ')
            gp = GaussianParameters(value, central, sigma=sigma)
        elif mode=='cov':
            covariance = Array("covariance", [
                                   [var_in[0], 0.0,       0.5],
                                   [0.0,       var_in[1], 1.0],
                                   [0.5,       1.0,       var_in[2]],
                                ],
                               mark='V',
                               mode='store_weak')
            gp = GaussianParameters(value, central, covariance=covariance)
        elif mode=='cov1d':
            covariance = Array("covariance", var_in, mark='diag(V)', mode='store_weak')
            gp = GaussianParameters(value, central, covariance=covariance)

    savegraph(graph, f"output/test_variables_00_{mode}.png")

    # import IPython; IPython.embed(colors='neutral')

