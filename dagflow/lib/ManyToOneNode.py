from ..input_extra import MissingInputAddOne
from ..nodes import FunctionNode
from ..typefunctions import (
    AllPositionals,
    check_has_inputs,
    check_inputs_equivalence,
    copy_input_edges_to_output,
    copy_input_shape_to_output,
    eval_output_dtype,
)

from ..node import Node
from ..storage import NodeStorage
from multikeydict.nestedmkdict import walkitems

from typing import Tuple, Union, Optional, Any
from multikeydict.typing import KeyLike

class ManyToOneNode(FunctionNode):
    """
    The abstract node with only one output `result`,
    which is the result of some function on all the positional inputs
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault(
            "missing_input_handler", MissingInputAddOne(output_fmt="result")
        )
        super().__init__(*args, **kwargs)

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        check_has_inputs(self) # at least one input
        check_inputs_equivalence(self) # all the inputs are have same dd fields
        copy_input_shape_to_output(self, 0, "result") # copy shape to result
        copy_input_edges_to_output(self, 0, "result") # copy edges to result
        eval_output_dtype(self, AllPositionals, "result") # eval dtype of result

    @classmethod
    def replicate(
        cls,
        name: str,
        *args: Union[NodeStorage, Any],
        replicate: Tuple[KeyLike,...]=((),),
        **kwargs
    ) -> Tuple[Optional[Node], NodeStorage]:
        storage = NodeStorage(default_containers=True)
        nodes = storage.child(f'nodes')
        outputs = storage.child(f'outputs')

        for outkey in replicate:
            outname = (name,)+outkey
            instance = cls('.'.join(outname), **kwargs)
            nodes[outname] = instance

            outkeyset = frozenset(outkey)
            for arg in args:
                for inkey, output in walkitems(arg):
                    inkeyset = frozenset(inkey)
                    if inkey and not outkeyset.issubset(inkeyset):
                        continue

                    try:
                        output >> instance
                    except TypeError as e:
                        raise ConnectionError(f"Invalid >> types for {inkey}/{outkey}: {type(output)}/{type(instance)}") from e

            outputs[outname] = instance.outputs[0]

        NodeStorage.update_current(storage, strict=True)

        if len(outkey)==1:
            return instance, storage

        return None, storage
