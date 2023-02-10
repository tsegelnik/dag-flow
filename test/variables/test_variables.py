#!/usr/bin/env python

from dagflow.lib import Array
from dagflow.lib.NormalizeCorrelatedVars import NormalizeCorrelatedVars
from dagflow.variable import Parameters, GaussianParameters
from dagflow.graph import Graph
from dagflow.graphviz import savegraph

def test_variables_00_parameter() -> None:
    pass

def test_variables_00_variable(mode='cov') -> None:
    with Graph(close=True) as graph:
        if mode=='single':
            value   = Array("variable", [1.1], mode='store_weak')
            central = Array("central",  [1.0], mode='store_weak')
            sigma   = Array("central",  [1.0], mode='store_weak')
            gp = GaussianParameters(value, central, sigma=sigma)
        elif mode=='uncorr':
            value   = Array("variable", [1.1, 1.8, 5.0], mode='store_weak')
            central = Array("central",  [1.0, 2.0, 3.0], mode='store_weak')
            sigma   = Array("central",  [1.0, 0.5, 2.0], mode='store_weak')
            gp = GaussianParameters(value, central, sigma=sigma)
        elif mode=='cov':
            value   = Array("variable", [1.1, 1.8, 5.0], mode='store_weak')
            central = Array("central",  [1.0, 2.0, 3.0], mode='store_weak')
            covariance = Array("covariance", [
                                   [1.0, 0.0, 0.5],
                                   [0.0, 0.5, 1.0],
                                   [0.5, 1.0, 2.0],
                                ],
                               mode='store_weak')
            gp = GaussianParameters(value, central, covariance=covariance)

    savegraph(graph, f"output/test_variables_00_{mode}.png")

    # import IPython; IPython.embed(colors='neutral')

