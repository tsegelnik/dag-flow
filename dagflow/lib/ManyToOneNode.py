from typing import Any, Optional, Sequence, Tuple

from multikeydict.nestedmkdict import walkkeys
from multikeydict.typing import KeyLike, TupleKey, properkey

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
        kwargs.setdefault("missing_input_handler", MissingInputAdd(input_fmt=self._input_names()))
        super().__init__(*args, **kwargs)
        self._add_output(output_name)
        self._broadcastable = broadcastable

    @staticmethod
    def _input_names() -> Tuple[str, ...]:
        return ("result",)

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
        *args: NodeStorage | Any,
        replicate: Sequence[KeyLike] = ((),),
        replicate_inputs: Sequence[KeyLike] | None = None,
        **kwargs,
    ) -> Tuple[Optional[Node], NodeStorage]:
        if args and replicate_inputs is not None:
            raise RuntimeError(
                "ManyToOneNode.replicate can use either `args` or `replicate_inputs`"
            )

        if replicate_inputs:
            return cls.replicate_from_indices(
                name, replicate=replicate, replicate_inputs=replicate_inputs, **kwargs
            )

        return cls.replicate_from_args(name, *args, replicate=replicate, **kwargs)

    @classmethod
    def replicate_from_args(
        cls,
        name: str,
        *args: NodeStorage | Any,
        replicate: Sequence[KeyLike] = ((),),
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

        def fcn_outer_after(_):
            nonlocal outputs, outname, instance
            outputs[outname] = instance.outputs[0]

        from multikeydict.match import match_keys

        keys_left = tuple(tuple(walkkeys(arg)) for arg in args)
        match_keys(
            keys_left,
            replicate,
            fcn,
            fcn_outer_before=fcn_outer_before,
            fcn_outer_after=fcn_outer_after,
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
    ) -> Tuple[Optional[Node], NodeStorage]:
        storage = NodeStorage(default_containers=True)
        nodes = storage("nodes")
        outputs = storage("outputs")
        inputs = storage("inputs")

        if not replicate:
            raise RuntimeError("`replicate` tuple should have at least one item")

        instance = None
        outname = ""

        input_names = cls._input_names()

        fullname = properkey(fullname, sep=".")
        if len(input_names)==1:
            path, name = fullname[:-1], fullname[-1]
        else:
            path, name = fullname, fullname[-1]

        def fcn_outer_before(outkey: TupleKey):
            nonlocal outname, instance, nodes
            outname = fullname + outkey
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

        from multikeydict.match import match_keys

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
