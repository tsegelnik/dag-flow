from collections.abc import Sequence
from typing import Any, Optional

from multikeydict.typing import KeyLike

from ..inputhandler import MissingInputAddPair
from ..node import Node
from ..nodes import FunctionNode
from ..storage import NodeStorage


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
            assign_outputs_axes_from_inputs,
            check_has_inputs,
            copy_from_input_to_output,
        )

        check_has_inputs(self)
        copy_from_input_to_output(self, slice(None), slice(None), edges=True, meshes=True)
        assign_outputs_axes_from_inputs(
            self, slice(None), slice(None), assign_meshes=True, ignore_assigned=True, ignore_Nd=True
        )

    @classmethod
    def replicate(
        cls,
        name: str,
        *args: NodeStorage | Any,
        replicate: Sequence[KeyLike] | None = None,
        **kwargs,
    ) -> tuple[Optional[Node], NodeStorage]:
        if args and replicate is not None:
            raise RuntimeError("OneToOneNode.replicate can use either `args` or `replicate`")

        if args:
            return cls.replicate_from_args(name, *args, **kwargs)

        if replicate is None:
            replicate = ((),)

        return cls.replicate_from_indices(name, replicate=replicate, **kwargs)

    @classmethod
    def replicate_from_indices(
        cls,
        name: str,
        replicate: Sequence[KeyLike] = ((),),
        **kwargs,
    ) -> tuple[Optional[Node], NodeStorage]:
        storage = NodeStorage(default_containers=True)
        nodes = storage("nodes")
        inputs = storage("inputs")
        outputs = storage("outputs")

        if not replicate:
            raise RuntimeError("`replicate` tuple should have at least one item")

        for outkey in replicate:
            outname = (name,) + outkey
            instance = cls(".".join(outname), **kwargs)
            nodes[outname] = instance
            instance()
            inputs[outname] = instance.inputs[0]
            outputs[outname] = instance.outputs[0]

        NodeStorage.update_current(storage, strict=True)

        if len(replicate) == 1:
            return instance, storage

        return None, storage

    @classmethod
    def replicate_from_args(
        cls,
        name: str,
        *args: NodeStorage | Any,
        **kwargs,
    ) -> tuple[Optional[Node], NodeStorage]:
        storage = NodeStorage(default_containers=True)
        nodes = storage("nodes")
        outputs = storage("outputs")

        from multikeydict.nestedmkdict import walkitems

        nobjects = 0
        for arg in args:
            for key, obj in walkitems(arg):
                outname = (name,) + key
                instance = cls(".".join(outname), **kwargs)
                nodes[outname] = instance
                obj >> instance  # pyright: ignore [reportUnusedExpression]
                outputs[outname] = instance.outputs[0]
                nobjects += 1

        NodeStorage.update_current(storage, strict=True)

        if nobjects == 1:
            return instance, storage

        return None, storage
