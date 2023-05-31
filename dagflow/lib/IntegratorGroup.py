from .Integrator import Integrator
from .IntegratorSampler import IntegratorSampler
from ..meta_node import MetaNode

from typing import Mapping, TYPE_CHECKING
if TYPE_CHECKING:
    from ..node import Node

from ..meta_node import MetaNode
from .Integrator import Integrator
from .IntegratorSampler import IntegratorSampler


class IntegratorGroup(MetaNode):
    __slots__ = ("_sampler")

    _sampler: "Node"
    def __init__(self, mode: str, *, dropdim: bool=True, labels: Mapping={}):
        super().__init__()
        self._init_sampler(mode, "sampler", labels.get("sampler", {}))
        self._add_integrator("integrator", labels.get("integrator", {}), dropdim=dropdim)

    def _init_sampler(self, mode: str, name: str="sampler", label: Mapping={}):
        self._sampler = IntegratorSampler(name, mode=mode, label=label)

        self._add_node(
            self._sampler,
            kw_inputs=["ordersX"],
            kw_inputs_optional=["ordersY"],
            kw_outputs=["x"],
            kw_outputs_optional=["y"],
            merge_inputs=["ordersX", "ordersY"]
        )

    def _add_integrator(self, name: str="integrator", label: Mapping={}, *, dropdim: bool):
        integrator = Integrator(name, dropdim=dropdim, label=label)
        if self._sampler.mode=="2d":
            integrator("ordersY")
        self._sampler.outputs["weights"] >> integrator("weights")

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
