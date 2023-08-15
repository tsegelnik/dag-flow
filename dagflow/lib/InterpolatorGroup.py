from typing import Literal, Mapping, Tuple
from multikeydict.typing import KeyLike

from ..meta_node import MetaNode
from ..node import Node
from ..storage import NodeStorage
from .Interpolator import Interpolator
from .SegmentIndex import SegmentIndex

class InterpolatorGroup(MetaNode):
    __slots__ = ("_indexer",)

    _indexer: "Node"
    def __init__(
            self,
            *,
            bare: bool=False,
            labels: Mapping={},
            **kwargs
            ):
        super().__init__()
        if bare:
            return

        self._init_indexer("indexer", label=labels.get("indexer", {}))
        self._add_interpolator("interpolator", label=labels.get("interpolator", {}), **kwargs)

    def _init_indexer(
        self,
        name: str,
        *,
        label={}
    ):
        self._indexer = SegmentIndex(name, label=label)
        self._add_node(
                self._indexer,
                kw_inputs=["coarse", "fine"],
                merge_inputs=["coarse", "fine"],
                )

    def _add_interpolator(
        self,
        name: str,
        method: Literal["linear", "log", "logx", "exp"] = "linear",
        *,
        positionals: bool=True,
        tolerance: float = 1e-10,
        underflow: Literal["constant", "nearestedge", "extrapolate"] = "extrapolate",
        overflow: Literal["constant", "nearestedge", "extrapolate"] = "extrapolate",
        fillvalue: float = 0.0,
        label={},
    ) -> Interpolator:
        interpolator = Interpolator(
                name,
                method=method,
                tolerance=tolerance,
                underflow=underflow,
                overflow=overflow,
                fillvalue=fillvalue,
                label=label,
                )
        self._indexer.outputs["indices"] >> interpolator("indices")

        self._add_node(
                interpolator,
                kw_inputs=["coarse", "fine"],
                merge_inputs=["coarse", "fine"],
                inputs_pos=positionals,
                outputs_pos=positionals,
                #missing_inputs=True,
                #also_missing_outputs=True,
                )
        return interpolator

    @classmethod
    def replicate(
        cls,
        method: Literal["linear", "log", "logx", "exp"] = "linear",
        name_indexer: str="indexer",
        name_interpolator: str="interpolator",
        labels: Mapping={},
        *,
        replicate: Tuple[KeyLike,...]=((),),
        **kwargs
    ) -> Tuple["InterpolatorGroup", "NodeStorage"]:
        storage = NodeStorage(default_containers=True)
        nodes = storage('nodes')
        inputs = storage('inputs')
        outputs = storage('outputs')

        interpolators = cls(bare=True)

        interpolators._init_indexer(name_indexer, label=labels.get("indexer", {}))
        label_int = labels.get("interpolator", {})
        for key in replicate:
            if isinstance(key, str):
                key = key,
            name = ".".join((name_interpolator,) + key)
            interpolator = interpolators._add_interpolator(name, method, label=label_int, positionals=False, **kwargs)
            nodes.child(name_interpolator)[key] = interpolator
            inputs.child(name_interpolator)[("ycoarse",) + key] = interpolator.inputs['y']
            outputs.child(name_interpolator)[key] = interpolator.outputs[0]

        inputs.child(name_interpolator)["xcoarse"] = interpolators.inputs['coarse']
        inputs.child(name_interpolator)["xfine"] = interpolators.inputs['fine']

        NodeStorage.update_current(storage, strict=True)

        return interpolators, storage
