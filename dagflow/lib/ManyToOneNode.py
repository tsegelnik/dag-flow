from ..input_extra import MissingInputAdd
from ..nodes import FunctionNode
from ..typefunctions import (
    AllPositionals,
    check_has_inputs,
    check_inputs_equivalence,
    copy_from_input_to_output,
    eval_output_dtype,
    AllPositionals
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
    __slots__ = ('_broadcastable',)
    _broadcastable: bool

    def __init__(self, *args, broadcastable: bool=False, **kwargs):
        kwargs.setdefault(
            "missing_input_handler", MissingInputAdd()
        )
        super().__init__(*args, **kwargs)
        self._add_output("result")
        self._broadcastable = broadcastable

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        check_has_inputs(self) # at least one input
        check_inputs_equivalence(self, broadcastable=self._broadcastable) # all the inputs are have same dd fields
        copy_from_input_to_output(
            self,
            AllPositionals,
            "result",
            prefer_largest_input=self._broadcastable,
            prefer_input_with_edges=True
        ) # copy shape to result
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
        nodes = storage('nodes')
        outputs = storage('outputs')

        for outkey in replicate:
            if isinstance(outkey, str):
                outkey = outkey,
            outname = (name,)+outkey
            instance = cls('.'.join(outname), **kwargs)
            nodes[outname] = instance

            outkeyset = frozenset(outkey)
            for arg in args:
                for inkey, output in walkitems(arg):
                    inkeyset = frozenset(inkey)
                    if inkey and not outkeyset.issubset(inkeyset):
                        if inkeyset.intersection(outkeyset):
                            raise ConnectionError(f"Unsupported LHS key {inkey}. RHS key is {outkey}")
                        continue

                    try:
                        output >> instance
                    except TypeError as e:
                        raise ConnectionError(f"Invalid >> types for {inkey}/{outkey}: {type(output)}/{type(instance)}") from e

            outputs[outname] = instance.outputs[0]

        NodeStorage.update_current(storage, strict=True)

        if len(replicate)==1:
            return instance, storage

        return None, storage
