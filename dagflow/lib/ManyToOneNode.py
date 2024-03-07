from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from typing import TYPE_CHECKING

from multikeydict.typing import properkey

from ..inputhandler import MissingInputAddOne
from ..node import Node
from ..nodes import FunctionNode
from ..storage import NodeStorage

if TYPE_CHECKING:
    from multikeydict.typing import KeyLike, TupleKey


class ManyToOneNode(FunctionNode):
    """
    The abstract node with only one output `result`,
    which is the result of some function of all the positional inputs
    """

    __slots__ = ("_broadcastable",)

    _broadcastable: bool

    def __init__(self, *args, broadcastable: bool = False, output_name: str = "result", **kwargs):
        kwargs.setdefault(
            "missing_input_handler", MissingInputAddOne(input_fmt=self._input_names())
        )
        super().__init__(*args, **kwargs)
        self._add_output(output_name)
        self._broadcastable = broadcastable

    @staticmethod
    def _input_names() -> tuple[str, ...]:
        return ("input",)

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        from ..typefunctions import (
            AllPositionals,
            check_has_inputs,
            check_inputs_equivalence,
            copy_from_input_to_output,
            eval_output_dtype,
        )

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
        *args: NodeStorage | Any,
        name: str,
        replicate: Sequence[KeyLike] = ((),),
        replicate_inputs: Sequence[KeyLike] | None = None,
        **kwargs,
    ) -> tuple[Node | None, NodeStorage]:
        if args and replicate_inputs is not None:
            raise RuntimeError(
                "ManyToOneNode.replicate can use either `args` or `replicate_inputs`"
            )

        if args:
            return cls.replicate_from_args(name, *args, replicate=replicate, **kwargs)

        if replicate_inputs:
            return cls.replicate_from_indices(
                name, replicate=replicate, replicate_inputs=replicate_inputs, **kwargs
            )

        return cls.replicate_from_indices(name, replicate=replicate, **kwargs)

    @classmethod
    def replicate_from_args(
        cls,
        fullname: str,
        *args: NodeStorage | Any,
        replicate: Sequence[KeyLike] = ((),),
        allow_skip_inputs: bool = False,
        **kwargs,
    ) -> tuple[Node | None, NodeStorage]:
        storage = NodeStorage(default_containers=True)
        nodes = storage("nodes")
        outputs = storage("outputs")

        if not replicate:
            raise RuntimeError("`replicate` tuple should have at least one item")

        instance = None
        outname = ""

        path = properkey(fullname, sep=".")

        def fcn_outer_before(outkey: TupleKey):
            nonlocal outname, instance
            outname = path + outkey
            instance = cls(".".join(outname), **kwargs)
            nodes[outname] = instance

        def fcn(i: int, inkey: TupleKey, outkey: TupleKey):
            container = args[i]
            output = container[inkey] if inkey else container
            try:
                output >> instance  # pyright: ignore [reportUnusedExpression]
            except TypeError as e:
                raise ConnectionError(
                    f"Invalid >> types for {inkey}/{outkey}: {type(output)}/{type(instance)}"
                ) from e

        def fcn_outer_after(_):
            outputs[outname] = instance.outputs[0]

        from multikeydict.tools import match_keys
        from multikeydict.nestedmkdict import walkkeys

        keys_left = tuple(tuple(walkkeys(arg)) for arg in args)
        match_keys(
            keys_left,
            replicate,
            fcn,
            fcn_outer_before=fcn_outer_before,
            fcn_outer_after=fcn_outer_after,
            require_all_left_keys_processed=not allow_skip_inputs,
        )

        NodeStorage.update_current(storage, strict=True)

        if len(replicate) == 1:
            return instance, storage  # pyright: ignore [reportUnboundVariable]

        return None, storage

    @classmethod
    def replicate_from_indices(
        cls,
        fullname: str,
        *,
        replicate: Sequence[KeyLike] = ((),),
        replicate_inputs: Sequence[KeyLike] = ((),),
        **kwargs,
    ) -> tuple[Node | None, NodeStorage]:
        storage = NodeStorage(default_containers=True)
        nodes = storage("nodes")
        outputs = storage("outputs")
        inputs = storage("inputs")

        if not replicate:
            raise RuntimeError("`replicate` tuple should have at least one item")

        instance = None
        outname = ""

        input_names = cls._input_names()

        path = properkey(fullname, sep=".")

        def fcn_outer_before(outkey: TupleKey):
            nonlocal outname, instance, nodes
            outname = path + outkey
            instance = cls(".".join(outname), **kwargs)
            nodes[outname] = instance

        def fcn(iarg: int, inkey: TupleKey, outkey: TupleKey):
            nonlocal inputs, instance, input_names
            for iname in input_names:
                input = instance()
                inputs[path + (iname,) + inkey] = input

        def fcn_outer_after(_):
            nonlocal outputs, outname, instance
            outputs[outname] = instance.outputs[0]

        from multikeydict.tools import match_keys

        match_keys(
            (replicate_inputs,),
            replicate,
            fcn,
            fcn_outer_before=fcn_outer_before,
            fcn_outer_after=fcn_outer_after,
        )

        NodeStorage.update_current(storage, strict=True)

        if len(replicate) == 1:
            return instance, storage  # pyright: ignore [reportUnboundVariable]

        return None, storage
