from .IBDXsecO1 import IBDXsecO1
from .EeToEnu import EeToEnu
from .Jacobian_dEnu_dEe import Jacobian_dEnu_dEe

from dagflow.meta_node import MetaNode

def IBDXsecO1Group(*, labels: dict={}):
    lenu = labels.setdefault('enu', {})
    lenu.setdefault('text', r'Eν, MeV')
    lenu.setdefault('latex', r'$E_{\nu}$, MeV')
    lenu.setdefault('axis', r'$E_{\nu}$, MeV')

    ljacobian = labels.setdefault('jacobian', {})
    ljacobian.setdefault('text', r'Energy conversion Jacobian dEν/dEdep')
    ljacobian.setdefault('latex', r'$dE_{\nu}/dE_{\mathrm dep}$')
    ljacobian.setdefault('axis', r'$dE_{\nu}/dE_{\mathrm dep}$')

    lxsec = labels.setdefault('xsec', {})
    lxsec.setdefault('text', r'IBD cross section σ(Eν,cosθ), cm⁻²')
    lxsec.setdefault('latex', r'IBD cross section $\sigma(E_{\nu}, \cos\theta)$, cm$^{-2}$')
    lxsec.setdefault('axis', r'$\sigma(E_{\nu}, \cos\theta)$, cm$^{-2}$')

    ibdxsec_ee = IBDXsecO1('ibd_Ee', label=lxsec)
    eetoenu = EeToEnu('Enu', label=lenu)
    jacobian = Jacobian_dEnu_dEe('dEν/dEe', label=ljacobian)

    eetoenu.outputs['result'] >> (jacobian.inputs['enu'], ibdxsec_ee.inputs['enu'])

    inputs_common = ['ElectronMass', 'ProtonMass', 'NeutronMass']
    inputs_ibd = inputs_common+[ 'NeutronLifeTime', 'PhaseSpaceFactor', 'g', 'f', 'f2' ]
    merge_inputs = ['ee', 'costheta']+inputs_common
    ibd = MetaNode()
    ibd.add_node(ibdxsec_ee, kw_inputs=['costheta']+inputs_ibd,                 merge_inputs=merge_inputs, outputs_pos=True)
    ibd.add_node(eetoenu,    kw_inputs=['ee', 'costheta']+inputs_common,        merge_inputs=merge_inputs,
                 kw_outputs={'result': 'enu'})
    ibd.add_node(jacobian,   kw_inputs=['enu', 'ee', 'costheta']+inputs_common, merge_inputs=merge_inputs,
                 kw_outputs={'result': 'jacobian'})
    ibd.inputs.make_positionals('ee', 'costheta')

    return ibd

