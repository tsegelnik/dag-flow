from typing import Any, Optional, Tuple, Union

from multikeydict.nestedmkdict import walkkeys
from multikeydict.typing import KeyLike, TupleKey

from ..inputhandler import MissingInputAdd
from ..node import Node
from ..nodes import FunctionNode
from ..storage import NodeStorage
from ..typefunctions import (
    AllPositionals,
    check_has_inputs,
    check_inputs_equivalence,
    copy_from_input_to_output,
    eval_output_dtype,
)


class ManyToOneNode(FunctionNode):
    """
    The abstract node with only one output `result`,
    which is the result of some function of all the positional inputs
    """

    __slots__ = ("_broadcastable",)
    _broadcastable: bool

    def __init__(self, *args, broadcastable: bool = False, output_name: str = "result", **kwargs):
        kwargs.setdefault("missing_input_handler", MissingInputAdd())
        super().__init__(*args, **kwargs)
        self._add_output(output_name)
        self._broadcastable = broadcastable

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        check_has_inputs(self)  # at least one input
        check_inputs_equivalence(
            self, broadcastable=self._broadcastable
        )  # all the inputs should have same dd fields
        copy_from_input_to_output(
            self,
            AllPositionals,
            "result",
            prefer_largest_input=self._broadcastable,
            prefer_input_with_edges=True,
        )  # copy shape to result
        eval_output_dtype(self, AllPositionals, "result")  # eval dtype of result

    @classmethod
    def replicate(
        cls,
        name: str,
        *args: Union[NodeStorage, Any],
        replicate: Tuple[KeyLike, ...] = ((),),
        **kwargs,
    ) -> Tuple[Optional[Node], NodeStorage]:
        storage = NodeStorage(default_containers=True)
        nodes = storage("nodes")
        outputs = storage("outputs")

        if not replicate:
            raise RuntimeError("`replicate` tuple should have at least one item")

        instance = None
        outname = ""

        def fcn_outer_before(outkey: TupleKey):
            nonlocal outname, instance, nodes
            outname = (name,) + outkey
            instance = cls(".".join(outname), **kwargs)
            nodes[outname] = instance

        def fcn(i: int, inkey: TupleKey, outkey: TupleKey):
            nonlocal args, instance
            container = args[i]
            output = container[inkey] if inkey else container
            try:
                output >> instance  # pyright: ignore [reportUnusedExpression]
            except TypeError as e:
                raise ConnectionError(
                    f"Invalid >> types for {inkey}/{outkey}: {type(output)}/{type(instance)}"
                ) from e

        def fcn_outer_after(outkey: TupleKey):
            nonlocal outputs, outname, instance
            outputs[outname] = instance.outputs[0]

        from multikeydict.match import match_keys

        keys_left = tuple(tuple(walkkeys(arg)) for arg in args)
        match_keys(
            keys_left, replicate, fcn, fcn_outer_before=fcn_outer_before, fcn_outer_after=fcn_outer_after
        )

        NodeStorage.update_current(storage, strict=True)

        if len(replicate) == 1:
            return instance, storage  # pyright: ignore [reportUnboundVariable]

        return None, storage
