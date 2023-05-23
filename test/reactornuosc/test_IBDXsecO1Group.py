#!/usr/bin/env python
from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib.Array import Array
from dagflow.bundles.load_parameters import load_parameters
from numpy import linspace, meshgrid, meshgrid

from reactornueosc.IBDXsecO1Group import IBDXsecO1Group
from dagflow.plot import plot_auto

from matplotlib.pyplot import close, subplots

def test_IBDXsecO1Group(debug_graph, testname):
    data = {
            'format': 'value',
            'state': 'fixed',
            'parameters': {
                'NeutronLifeTime': 879.4,        # s,   page 165
                'NeutronMass':     939.565413,   # MeV, page 165
                'ProtonMass':      938.272081,   # MeV, page 163
                'ElectronMass':    0.5109989461, # MeV, page 16
                'PhaseSpaceFactor': 1.71465,
                'g':                1.2701,
                'f':                1.0,
                'f2':               3.706,
                },
            'labels': {
                'NeutronLifeTime': 'neutron lifetime, s (PDG2014)',
                'NeutronMass': 'neutron mass, MeV (PDG2012)',
                'ProtonMass': 'proton mass, MeV (PDG2012)',
                'ElectronMass': 'electron mass, MeV (PDG2012)',
                'PhaseSpaceFactor': "IBD phase space factor",
                'f': "vector coupling constant f",
                'g': "axial-vector coupling constant g",
                'f2': "anomalous nucleon isovector magnetic moment f₂",
                }
            }

    enu1 = linspace(1, 12.0, 111)
    ee1 = enu1.copy()
    ctheta1 = linspace(-1, 1, 5)
    enu2, ctheta2 = meshgrid(enu1, ctheta1, indexing='ij')
    ee2, _ = meshgrid(ee1, ctheta1, indexing='ij')

    with Graph(debug=debug_graph, close=True) as graph:
        storage = load_parameters(data)

        ee = Array(
            'ee', ee2,
            label={'axis': r'$E_{\rm pos}$, MeV'}
        )
        ctheta = Array('ctheta', ctheta2,
                       label={'axis': r'$\cos\theta$'}
                       )

        ibdxsec = IBDXsecO1Group()

        ibdxsec << storage('parameter.constant')
        ee >> ibdxsec.inputs['ee']
        ctheta >> ibdxsec.inputs['costheta']
        ibdxsec.print(recursive=True)

    csc_ee = ibdxsec.get_data()

    show = True
    close = not show
    from mpl_toolkits.mplot3d import axes3d # accessed implicitly in `subplots()`
    subplots(1, 1, subplot_kw={'projection': '3d'})
    plot_auto(ibdxsec, mode='surface', cmap=True, colorbar=True,
              show=False, close=close, save=f"output/{testname}_xsec_surf.pdf")

    subplots(1, 1, subplot_kw={'projection': '3d'})
    plot_auto(ibdxsec, mode='wireframe', cmap=True, colorbar=True,
              show=False, close=close, save=f"output/{testname}_xsec_wirec.pdf")

    subplots(1, 1, subplot_kw={'projection': '3d'})
    plot_auto(ibdxsec, mode='wireframe',
              show=False, close=close, save=f"output/{testname}_xsec_wire.pdf")

    subplots(1, 1)
    plot_auto(ibdxsec, mode='pcolormesh', colorbar=True,
              show=False, close=close, save=f"output/{testname}_xsec_mesh.pdf")

    subplots(1, 1)
    plot_auto(ibdxsec.outputs['enu'], mode='pcolormesh', colorbar=True,
              show=False, close=close, save=f"output/{testname}_enu_mesh.pdf")

    subplots(1, 1, subplot_kw={'projection': '3d'})
    plot_auto(ibdxsec.outputs['enu'], mode='surface', cmap=True, colorbar=True,
              show=False, close=close, save=f"output/{testname}_enu_surf.pdf")

    subplots(1, 1)
    plot_auto(ibdxsec.outputs['jacobian'], mode='pcolormesh', colorbar=True,
              show=False, close=close, save=f"output/{testname}_jac_mesh.pdf")

    subplots(1, 1, subplot_kw={'projection': '3d'})
    plot_auto(ibdxsec.outputs['jacobian'], mode='surface', cmap=True, colorbar=True,
              show=show, close=True, save=f"output/{testname}_jac_surf.pdf")

    savegraph(graph, f"output/{testname}.pdf")

