from ..input_extra import MissingInputAddPair
from ..nodes import FunctionNode
from ..node import Node
from ..storage import NodeStorage

from multikeydict.typing import KeyLike

from typing import Tuple, Optional

class OneToOneNode(FunctionNode):
    """
    The abstract node with an output for every positional input
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("missing_input_handler", MissingInputAddPair())
        super().__init__(*args, **kwargs)

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        from ..typefunctions import (
            check_has_inputs,
            copy_from_input_to_output,
            assign_outputs_axes_from_inputs
        )
        check_has_inputs(self)
        copy_from_input_to_output(self, slice(None), slice(None), edges=True, meshes=True)
        assign_outputs_axes_from_inputs(self, slice(None), slice(None), assign_meshes=True, ignore_assigned=True, ignore_Nd=True)

    @classmethod
    def replicate(
        cls,
        name: str,
        replicate: Tuple[KeyLike,...]=((),),
        **kwargs
    ) -> Tuple[Optional[Node], NodeStorage]:
        storage = NodeStorage(default_containers=True)
        nodes = storage('nodes')
        inputs = storage('inputs')
        outputs = storage('outputs')

        for outkey in replicate:
            outname = (name,)+outkey
            instance = cls('.'.join(outname), **kwargs)
            nodes[outname] = instance
            instance()
            inputs[outname] = instance.inputs[0]
            outputs[outname] = instance.outputs[0]

        NodeStorage.update_current(storage, strict=True)

        if len(outkey)==1:
            return instance, storage

        return None, storage
