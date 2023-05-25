from .Integrator import Integrator
from .IntegratorSampler import IntegratorSampler
from ..meta_node import MetaNode

from typing import Mapping

def IntegratorGroup(mode: str, *, labels: Mapping={}) -> MetaNode:
    sampler = IntegratorSampler("sampler", mode=mode, label=labels.get('sampler', {}))
    integrator = Integrator("integrator", label=labels.get('integrator', {}))
    sampler.outputs["weights"] >> integrator("weights")

    metaint = MetaNode()
    metaint._add_node(sampler, kw_inputs=['ordersX'], kw_inputs_optional=['ordersY'], kw_outputs=['x'], merge_inputs=['ordersX', 'ordersY'])
    metaint._add_node(integrator, kw_inputs=['ordersX'], merge_inputs=['ordersX', 'ordersY'],
                     inputs_pos=True, outputs_pos=True, missing_inputs=True, also_missing_outputs=True)

    return metaint
