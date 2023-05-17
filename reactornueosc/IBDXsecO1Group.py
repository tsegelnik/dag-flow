from .IBDXsecO1 import IBDXsecO1
from .EeToEnu import EeToEnu
from .Jacobian_dEnu_dEe import Jacobian_dEnu_dEe

from dagflow.meta_node import MetaNode

def IBDXsecO1Group(*, labels: dict={}):
    ibdxsec_ee = IBDXsecO1('ibd_Ee', label=labels.get('xsec', {}))
    eetoenu = EeToEnu('Enu', label=labels.get('enu', {}))
    jacobian = Jacobian_dEnu_dEe('dEν/dEe', label=labels.get('jacobian', {}))

    eetoenu.outputs['result'] >> (jacobian.inputs['enu'], ibdxsec_ee.inputs['enu'])

    inputs_common = ['ElectronMass', 'ProtonMass', 'NeutronMass']
    inputs_ibd = inputs_common+[ 'NeutronLifeTime', 'PhaseSpaceFactor', 'g', 'f', 'f2' ]
    merge_inputs = ['ee', 'costheta']+inputs_common
    ibd = MetaNode()
    ibd.add_node(ibdxsec_ee, kw_inputs=['costheta']+inputs_ibd,                 merge_inputs=merge_inputs, outputs_pos=True)
    ibd.add_node(eetoenu,    kw_inputs=['ee', 'costheta']+inputs_common,        merge_inputs=merge_inputs,
                 kw_outputs={'result': 'enu'})
    ibd.add_node(jacobian,   kw_inputs=['enu', 'ee', 'costheta']+inputs_common[:-1], merge_inputs=merge_inputs[:-1],
                 kw_outputs={'result': 'jacobian'})
    ibd.inputs.make_positionals('ee', 'costheta')

    return ibd

