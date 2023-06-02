from multikeydict.typing import KeyLike

from .Integrator import Integrator
from .IntegratorSampler import IntegratorSampler, ModeType
from ..meta_node import MetaNode

from typing import Mapping, TYPE_CHECKING, Tuple, Union
if TYPE_CHECKING:
    from ..node import Node
    from ..storage import NodeStorage

from ..meta_node import MetaNode
from .Integrator import Integrator
from .IntegratorSampler import IntegratorSampler


class IntegratorGroup(MetaNode):
    __slots__ = ("_sampler")

    _sampler: "Node"
    def __init__(
        self,
        mode: ModeType,
        *,
        bare: bool=False,
        dropdim: bool=True,
        labels: Mapping={}
    ):
        super().__init__()
        if bare:
            return

        self._init_sampler(mode, "sampler", labels.get("sampler", {}))
        self._add_integrator("integrator", labels.get("integrator", {}), dropdim=dropdim)

    def _init_sampler(self, mode: ModeType, name: str="sampler", label: Mapping={}):
        self._sampler = IntegratorSampler(name, mode=mode, label=label)

        self._add_node(
            self._sampler,
            kw_inputs=["ordersX"],
            kw_inputs_optional=["ordersY"],
            kw_outputs=["x"],
            kw_outputs_optional=["y"],
            merge_inputs=["ordersX", "ordersY"]
        )

    def _add_integrator(
        self,
        name: str="integrator",
        label: Mapping={},
        *,
        positionals: bool=True,
        dropdim: bool
    ) -> Integrator:
        integrator = Integrator(name, dropdim=dropdim, label=label)
        if self._sampler.mode=="2d":
            integrator("ordersY")
        self._sampler.outputs["weights"] >> integrator("weights")

        self._add_node(
            integrator,
            kw_inputs=["ordersX"],
            kw_inputs_optional=["ordersY"],
            merge_inputs=["ordersX", "ordersY"],
            inputs_pos=positionals,
            outputs_pos=positionals,
            missing_inputs=True,
            also_missing_outputs=True,
        )

        return integrator

    @classmethod
    def replicate(
        cls,
        mode: ModeType,
        name_sampler: str="sampler",
        name_integrator: str="integrator",
        labels: Mapping={},
        *,
        replicate: Tuple[KeyLike,...]=((),),
        dropdim: bool=True
    ) -> Union["IntegratorGroup", "NodeStorage"]:
        from ..storage import NodeStorage
        storage = NodeStorage()
        nodes = storage.child('nodes')
        inputs = storage.child('inputs')
        outputs = storage.child('outputs')

        integrators = cls(mode, bare=True)

        integrators._init_sampler(mode, name_sampler, labels.get("sampler", {}))
        label_int = labels.get("integrator", {})
        for key in replicate:
            name = ".".join((name_integrator,) + key)
            integrator = integrators._add_integrator(name, label_int, positionals=False, dropdim=dropdim)
            integrator()
            nodes.child(name_integrator)[key] = integrator
            inputs.child(name_integrator)[key] = integrator.inputs[0]
            outputs.child(name_integrator)[key] = integrator.outputs[0]

        if (common_storage := NodeStorage.current()) is not None:
            common_storage^=storage

        return integrators, storage
