from .Integrator import Integrator
from .IntegratorSampler import IntegratorSampler
from ..meta_node import MetaNode

from typing import Mapping

def IntegratorGroup(mode: str, *, labels: Mapping={}) -> MetaNode:
    sampler = IntegratorSampler("sampler", mode=mode, label=labels.get('sampler', {}))
    integrator = Integrator("integrator", label=labels.get('integrator', {}))
    sampler.outputs["weights"] >> integrator("weights")

    metaint = MetaNode()
    metaint.add_node(sampler, kw_inputs=['ordersX'], kw_outputs=['x'], merge_inputs=['ordersX'])
    metaint.add_node(integrator, kw_inputs=['ordersX'], merge_inputs=['ordersX'],
                     inputs_pos=True, outputs_pos=True, missing_inputs=True, also_missing_outputs=True)

    return metaint
