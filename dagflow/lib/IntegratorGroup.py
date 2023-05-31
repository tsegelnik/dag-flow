from typing import Mapping

from ..meta_node import MetaNode
from .Integrator import Integrator
from .IntegratorSampler import IntegratorSampler


class IntegratorGroup(MetaNode):
    def __init__(self, mode: str, dropdim: bool = True, *, labels: Mapping = {}):
        super().__init__()

        sampler = IntegratorSampler(
            "sampler",
            mode=mode,
            label=labels.get("sampler", {}),
        )
        integrator = Integrator(
            "integrator", dropdim=dropdim, label=labels.get("integrator", {})
        )
        sampler.outputs["weights"] >> integrator("weights")

        if mode == "2d":
            integrator("ordersY")

        self._add_node(
            sampler,
            kw_inputs=["ordersX"],
            kw_inputs_optional=["ordersY"],
            kw_outputs=["x"],
            kw_outputs_optional=["y"],
            merge_inputs=["ordersX", "ordersY"],
        )
        self._add_node(
            integrator,
            kw_inputs=["ordersX"],
            kw_inputs_optional=["ordersY"],
            merge_inputs=["ordersX", "ordersY"],
            inputs_pos=True,
            outputs_pos=True,
            missing_inputs=True,
            also_missing_outputs=True,
        )
