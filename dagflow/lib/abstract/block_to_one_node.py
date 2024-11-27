from __future__ import annotations

from typing import TYPE_CHECKING

from multikeydict.typing import properkey

from ...core.input_handler import MissingInputAddEach
from ...core.node import Node
from ...core.storage import NodeStorage
from ...core.type_functions import (
    AllPositionals,
    check_node_has_inputs,
    check_inputs_equivalence,
    copy_from_inputs_to_outputs,
    evaluate_dtype_of_outputs,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from multikeydict.typing import KeyLike, TupleKey


class BlockToOneNode(Node):
    """
    The abstract node with only one output per block of N inputs
    """

    __slots__ = ("_broadcastable",)

    _broadcastable: bool

    def __init__(self, *args, broadcastable: bool = False, output_name: str = "result", **kwargs):
        kwargs.setdefault(
            "missing_input_handler",
            MissingInputAddEach(input_fmt=self._input_names(), output_fmt=output_name),
        )
        super().__init__(*args, **kwargs)
        self._broadcastable = broadcastable

    @staticmethod
    def _input_names() -> tuple[str, ...]:
        return ("input",)

    @classmethod
    def _inputs_block_size(cls) -> int:
        return len(cls._input_names())

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        check_node_has_inputs(self)  # at least one input
        check_inputs_equivalence(
            self, broadcastable=self._broadcastable
        )  # all the inputs should have same dd fields
        n = self._inputs_block_size()
        copy_from_inputs_to_outputs(
            self,
            slice(0, None, n),
            AllPositionals,
            prefer_largest_input=self._broadcastable,
            prefer_input_with_edges=True,
        )  # copy shape to results
        evaluate_dtype_of_outputs(self, AllPositionals, AllPositionals)  # eval dtype of results

    @classmethod
    def replicate(
        cls,
        *args: NodeStorage | Any,
        name: str,
        replicate_outputs: Sequence[KeyLike] = ((),),
        replicate_inputs: Sequence[KeyLike] | None = None,
        **kwargs,
    ) -> tuple[Node | None, NodeStorage]:
        if args and replicate_inputs is not None:
            raise RuntimeError(
                "ManyToOneNode.replicate can use either `args` or `replicate_inputs`"
            )

        if args:
            return cls.replicate_from_args(
                name, *args, replicate_outputs=replicate_outputs, **kwargs
            )

        if replicate_inputs:
            return cls.replicate_from_indices(
                name,
                replicate_outputs=replicate_outputs,
                replicate_inputs=replicate_inputs,
                **kwargs,
            )

        return cls.replicate_from_indices(name, replicate_outputs=replicate_outputs, **kwargs)

    @classmethod
    def replicate_from_args(
        cls,
        fullname: str,
        *args: NodeStorage | Any,
        replicate_outputs: Sequence[KeyLike] = ((),),
        allow_skip_inputs: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> tuple[Node | None, NodeStorage]:
        storage = NodeStorage(default_containers=True)
        nodes = storage("nodes")
        outputs = storage("outputs")

        if not replicate_outputs:
            raise RuntimeError("`replicate_outputs` tuple should have at least one item")

        instance = None
        outname = ""

        path = properkey(fullname, sep=".")

        def fcn_outer_before(outkey: TupleKey):
            nonlocal outname, instance
            outname = path + outkey
            instance = cls(".".join(outname), **kwargs)
            nodes[outname] = instance

        def function(i: int, inkey: TupleKey, outkey: TupleKey):
            container = args[i]
            output = container[inkey] if inkey else container
            try:
                output >> instance  # pyright: ignore [reportUnusedExpression]
            except TypeError as e:
                raise ConnectionError(
                    f"Invalid >> types for {inkey}/{outkey}: {type(output)}/{type(instance)}"
                ) from e

        def fcn_outer_after(_):
            outputs[outname] = instance.outputs[-1]

        from multikeydict.nestedmkdict import walkkeys
        from multikeydict.tools import match_keys

        keys_left = tuple(tuple(walkkeys(arg)) for arg in args)
        match_keys(
            keys_left,
            replicate_outputs,
            function,
            fcn_outer_before=fcn_outer_before,
            fcn_outer_after=fcn_outer_after,
            require_all_left_keys_processed=not allow_skip_inputs,
            require_all_right_keys_processed=False,
        )

        NodeStorage.update_current(storage, strict=True, verbose=verbose)

        if len(replicate_outputs) == 1:
            return instance, storage  # pyright: ignore [reportUnboundVariable]

        return None, storage

    @classmethod
    def replicate_from_indices(
        cls,
        fullname: str,
        *,
        replicate_outputs: Sequence[KeyLike] = ((),),
        replicate_inputs: Sequence[KeyLike] | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> tuple[Node | None, NodeStorage]:
        storage = NodeStorage(default_containers=True)
        nodes = storage("nodes")
        outputs = storage("outputs")
        inputs = storage("inputs")

        if not replicate_outputs:
            raise RuntimeError("`replicate_outputs` tuple should have at least one item")
        if replicate_inputs is None:
            replicate_inputs = replicate_outputs
        elif not replicate_inputs:
            raise RuntimeError("`replicate_inputs` tuple should have at least one item")

        instance = None
        outname = ("",)

        path = properkey(fullname, sep=".")

        def fcn_outer_before(outkey: TupleKey):
            nonlocal outname, instance, nodes
            outname = path + outkey
            instance = cls(".".join(outname), **kwargs)
            nodes[outname] = instance

        def function(iarg: int, inkey: TupleKey, outkey: TupleKey):
            nonlocal inputs, instance
            input_names = instance._input_names()
            for iname in input_names:
                input = instance()
                inputs[path + (iname,) + inkey] = input

        def fcn_outer_after(outkey: TupleKey):
            nonlocal outname, instance
            outputs[outname] = instance.outputs[-1]

        from multikeydict.tools import match_keys

        match_keys(
            (replicate_inputs,),
            replicate_outputs,
            function,
            fcn_outer_before=fcn_outer_before,
            fcn_outer_after=fcn_outer_after,
        )

        NodeStorage.update_current(storage, strict=True, verbose=verbose)

        if len(replicate_outputs) == 1:
            return instance, storage  # pyright: ignore [reportUnboundVariable]

        return None, storage
