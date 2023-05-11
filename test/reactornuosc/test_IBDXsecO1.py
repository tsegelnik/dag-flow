#!/usr/bin/env python
from dagflow.exception import TypeFunctionError
from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib.Array import Array
from dagflow.lib.Integrator import Integrator
from dagflow.lib.IntegratorSampler import IntegratorSampler
from dagflow.lib.trigonometry import Cos, Sin
from dagflow.bundles.load_parameters import load_parameters
from numpy import allclose, linspace, arange, meshgrid, pi, vectorize, meshgrid
from pytest import mark, raises

from reactornueosc.IBDXsecO1 import IBDXsecO1

def test_IBDXsecO1(debug_graph, testname):
    data = {
            'format': 'value',
            'state': 'fixed',
            'parameters': {
                'NeutronLifeTime': 879.4,        # s,   page 165
                'NeutronMass':     939.565413,   # MeV, page 165
                'ProtonMass':      938.272081,   # MeV, page 163
                'ElectronMass':    0.5109989461, # MeV, page 16
                },
            'labels': {
                'NeutronLifeTime': 'neutron lifetime, s (PDG2014)',
                'NeutronMass': 'neutron mass, MeV (PDG2012)',
                'ProtonMass': 'proton mass, MeV (PDG2012)',
                'ElectronMass': 'electron mass, MeV (PDG2012)'
                }
            }
    enu1 = linspace(1, 12.0, 111)
    ctheta1 = linspace(-1, 1, 5)
    enu2, ctheta2 = meshgrid(enu1, ctheta1, indexing='ij')

    with Graph(debug=debug_graph, close=True) as graph:
        storage = load_parameters(data)

        enu = Array('enu', enu2)
        ctheta = Array('ctheta', ctheta2)
        ibdxsec = IBDXsecO1('ibd')

        (enu, ctheta) >> ibdxsec
        ibdxsec << storage['parameter.constant']

    csc = ibdxsec.get_data()
    print(csc)

    #     npoints = 10
    #     edges = Array("edges", linspace(0, pi, npoints + 1))
    #     ordersX = Array("ordersX", [1000] * npoints, edges=edges["array"])
    #     A = Array("A", edges._data[:-1])
    #     B = Array("B", edges._data[1:])
    #     sampler = IntegratorSampler("sampler", mode="trap")
    #     integrator = Integrator("integrator")
    #     cosf = Cos("cos")
    #     sinf = Sin("sin")
    #     ordersX >> sampler("ordersX")
    #     sampler.outputs["x"] >> cosf
    #     A >> sinf
    #     B >> sinf
    #     sampler.outputs["weights"] >> integrator("weights")
    #     cosf.outputs[0] >> integrator
    #     ordersX >> integrator("ordersX")
    # res = sinf.outputs[1].data - sinf.outputs[0].data
    # assert allclose(integrator.outputs[0].data, res, atol=1e-2)
    # assert integrator.outputs[0].dd.axes_edges == [edges["array"]]
    savegraph(graph, f"output/{testname}.pdf")

