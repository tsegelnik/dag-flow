from __future__ import annotations

from typing import TYPE_CHECKING

from multikeydict.typing import properkey

from ...core.meta_node import MetaNode
from ...core.storage import NodeStorage
from .integrator_core import IntegratorCore
from .integrator_sampler import IntegratorSampler

if TYPE_CHECKING:
    from collections.abc import Mapping

    from multikeydict.typing import Key

    from ...core.node import Node
    from .integrator_sampler import ModeType


class Integrator(MetaNode):
    __slots__ = ("_sampler",)

    _sampler: Node

    def __init__(
        self,
        mode: ModeType,
        *,
        bare: bool = False,
        dropdim: bool = True,
        labels: Mapping = {},
    ):
        super().__init__()
        if bare:
            return

        self._init_sampler(mode, "sampler", labels.get("sampler", {}))
        self._add_integrator("integrator", labels.get("integrator", {}), dropdim=dropdim)

    def _init_sampler(self, mode: ModeType, name: str = "sampler", label: Mapping = {}):
        self._sampler = IntegratorSampler(name, mode=mode, label=label)

        self._add_node(
            self._sampler,
            kw_inputs=["ordersX"],
            kw_inputs_optional=["ordersY"],
            kw_outputs=["x"],
            kw_outputs_optional=["y"],
            merge_inputs=["ordersX", "ordersY"],
            missing_inputs=True,
            also_missing_outputs=True,
        )

    def _add_integrator(
        self,
        name: str = "integrator",
        label: Mapping = {},
        *,
        positionals: bool = True,
        dropdim: bool,
    ) -> IntegratorCore:
        integrator = IntegratorCore(name, dropdim=dropdim, label=label)
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
        *,
        names: Mapping[str, str] = {
            "sampler": "sampler",
            "integrator": "integrator",
            "x": "mesh_x",
            "y": "mesh_y",
        },
        labels: Mapping = {},
        replicate_outputs: tuple[Key, ...] = ((),),
        single_node: bool = False,
        dropdim: bool = True,
    ) -> tuple["Integrator", "NodeStorage"]:
        storage = NodeStorage(default_containers=True)
        nodes = storage("nodes")
        inputs = storage("inputs")
        outputs = storage("outputs")

        integrators = cls(mode, bare=True)
        key_integrator = tuple(names.get("integrator", "integrator").split("."))
        key_sampler = names.get("sampler", "sampler").split(".")
        key_meta = f"{key_integrator[0]}_meta".split(".")

        nodes[key_meta] = integrators

        integrators._init_sampler(mode, names.get("sampler", "sampler"), labels.get("sampler", {}))
        outputs[key_sampler + names.get("x", "x").split(".")] = integrators._sampler.outputs["x"]
        outputs[key_sampler + names.get("y", "y").split(".")] = integrators._sampler.outputs["y"]
        nodes[key_sampler] = integrators._sampler

        label_int = labels.get("integrator", {})
        integrator = None
        need_new_instance = not single_node
        for key in replicate_outputs:
            key = properkey(key)
            name = ".".join(key_integrator + key)

            if need_new_instance:
                integrator = integrators._add_integrator(
                    name, label_int, positionals=False, dropdim=dropdim
                )
                nodes[key_integrator + key] = integrator
            elif integrator is None:
                integrator = integrators._add_integrator(
                    name, label_int, positionals=False, dropdim=dropdim
                )
                nodes[key_integrator] = integrator

            # NOTE: it is needed to create an input and add to the storage
            integrator()
            inputs[key_integrator + key] = integrator.inputs[-1]
            outputs[key_integrator + key] = integrator.outputs[-1]

        NodeStorage.update_current(storage, strict=True)

        return integrators, storage
