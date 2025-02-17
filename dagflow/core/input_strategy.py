from __future__ import annotations

from typing import TYPE_CHECKING

from dagflow.core.exception import InitializationError

from ..tools.formatter import Formattable, LimbNameFormatter, SimpleFormatter

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .input import Input
    from .meta_node import MetaNode
    from .node_base import NodeBase


class InputStrategyBase:
    """The base class for a behaviour when output is connecting to the node with missing input via >>/<<"""

    __slots__ = ("_node", "_scope")

    _node: NodeBase
    _scope: int

    def __init__(self, node: NodeBase | None = None) -> None:
        self.node = node
        self.scope = 0

    @property
    def scope(self) -> int:
        return self._scope

    @scope.setter
    def scope(self, val) -> None:
        if isinstance(val, int):
            self._scope = val
        elif val is None:
            self._scope = 0
        else:
            raise InitializationError(f"Wrong type for scope={val}. Must be int.")

    @property
    def node(self) -> NodeBase:
        return self._node

    @node.setter
    def node(self, node) -> None:
        self._node = node

    def __call__(self, *args, **kwargs) -> None | Input:
        raise RuntimeError(
            f"Cannot create new inputs due to unimplemented strategy. Use one from {InputStrateges}"
        )


class AddNewInput(InputStrategyBase):
    """Adds an input for each output in >> operator"""

    __slots__ = ("input_fmt", "input_kws", "output_fmt", "output_kws")

    input_fmt: Formattable
    input_kws: dict
    output_fmt: Formattable
    output_kws: dict

    def __init__(
        self,
        node: NodeBase | None = None,
        *,
        input_fmt: str | Sequence[str] | LimbNameFormatter = SimpleFormatter("input", "_{:02d}"),
        input_kws: dict | None = None,
        output_fmt: str | Sequence[str] | LimbNameFormatter = SimpleFormatter("output", "_{:02d}"),
        output_kws: dict | None = None,
    ):
        if input_kws is None:
            input_kws = {}
        if output_kws is None:
            output_kws = {}
        super().__init__(node)
        self.input_kws = input_kws
        self.output_kws = output_kws
        self.input_fmt = LimbNameFormatter.from_value(input_fmt)
        self.output_fmt = LimbNameFormatter.from_value(output_fmt)

    def __call__(self, idx=None, scope=None, *, fmt: Formattable | None = None, **kwargs):
        kwargs_combined = dict(self.input_kws, **kwargs)
        if fmt is None:
            fmt = self.input_fmt
        if scope is None:
            scope = self.scope
        return self.node._add_input(
            fmt.format(idx if idx is not None else len(self.node.inputs)),
            **kwargs_combined,
        )


class AddNewInputAddNewOutput(AddNewInput):
    """
    Adds an input for each output in >> operator.
    Adds an output for each new input.
    """

    __slots__ = ()

    def __init__(self, node: NodeBase | None = None, **kwargs):
        super().__init__(node, **kwargs)

    def __call__(self, idx=None, scope=None, idx_out=None, **kwargs):
        if idx_out is None:
            idx_out = len(self.node.outputs)
        if scope is None:
            scope = self.scope
        out = self.node._add_output(self.output_fmt.format(idx_out), **self.output_kws)
        return super().__call__(idx, child_output=out, scope=scope, **kwargs)


class AddNewInputAddAndKeepSingleOutput(AddNewInput):
    """
    Adds an input for each output in >> operator.
    Adds only one output if needed.
    """

    __slots__ = ("add_child_output",)

    add_child_output: bool

    def __init__(self, node: NodeBase | None = None, *, add_child_output: bool = False, **kwargs):
        super().__init__(node, **kwargs)
        self.add_child_output = add_child_output

    def __call__(self, idx=None, scope=None, idx_out=None, **kwargs):
        if idx_out is not None:
            out = self.node._add_output(self.output_fmt.format(idx_out), **self.output_kws)
        elif (idx_out := len(self.node.outputs)) == 0:
            out = self.node._add_output(self.output_fmt.format(idx_out), **self.output_kws)
        else:
            out = self.node.outputs[-1]
        if scope is None:
            scope = self.scope
        if self.add_child_output:
            return super().__call__(idx, child_output=out, scope=scope)
        return super().__call__(idx, scope=scope, **kwargs)


class AddNewInputAddNewOutputForBlock(AddNewInput):
    """
    Adds an input for each block of outputs (for each >> operator).
    Adds an output for each block of outputs (for each >> operator).
    """

    __slots__ = ("add_child_output",)

    add_child_output: bool

    def __init__(self, node: NodeBase | None = None, *, add_child_output=False, **kwargs):
        super().__init__(node, **kwargs)
        self.add_child_output = add_child_output

    def __call__(self, idx=None, scope=None, **kwargs):
        if scope == self.scope or scope is None:
            try:
                out = self.node.outputs[-1]
            except IndexError:
                out = self.node._add_output(
                    self.output_fmt.format(len(self.node.outputs)),
                    **self.output_kws,
                )
        else:
            out = self.node._add_output(
                self.output_fmt.format(len(self.node.outputs)),
                **self.output_kws,
            )
            self._scope = scope
        if self.add_child_output:
            return super().__call__(idx, child_output=out, scope=scope)
        return super().__call__(idx, scope=scope, **kwargs)


class AddNewInputAddNewOutputForNInputs(AddNewInput):
    """
    Adds an input for each output in >> operator.
    Adds an output for each N inputs.
    """

    __slots__ = ("add_child_output", "n", "input_fmts", "starts_from_0")

    add_child_output: bool
    n: int

    def __init__(
        self,
        n: int,
        node: NodeBase | None = None,
        *,
        starts_from_0: bool = False,
        add_child_output=False,
        **kwargs,
    ):
        super().__init__(node, **kwargs)
        self.n = n
        if not isinstance(n, int) or n < 1:
            raise InitializationError(f"'n' must be int > 0, but given {n=}, {type(n)=}!")
        self.starts_from_0 = starts_from_0
        self._scope = 0
        self.add_child_output = add_child_output

    def __call__(self, idx=None, scope=None, **kwargs):
        out = None
        create_output = lambda: self.node._add_output(
            self.output_fmt.format(len(self.node.outputs)), **self.output_kws
        )
        if self.n == 1:
            out = create_output()
        elif self._scope % self.n == 0:
            if self._scope == 0 and self.starts_from_0 or self._scope != 0:
                out = create_output()
            else:
                self._scope += 1
        self._scope += 1
        if self.add_child_output:
            return super().__call__(idx, child_output=out, scope=scope)
        return super().__call__(idx, scope=scope)


class InheritInputStrategy(InputStrategyBase):
    """
    Inherit an input strategy from the source node. It is a special strategy for MetaNode.
    """

    __slots__ = ("_source_node", "_target_node", "_source_handler", "_inherit_outputs")
    _source_node: NodeBase
    _target_node: MetaNode
    _source_handler: InputStrategyBase
    _inherit_outputs: bool

    def __init__(
        self,
        source_node: NodeBase,
        target_node: MetaNode,
        *,
        inherit_outputs: bool = False,
    ):
        super().__init__()
        self._source_node = source_node
        self._target_node = target_node
        self._inherit_outputs = inherit_outputs

        try:
            self._source_handler = source_node._input_strategy
        except AttributeError as exc:
            raise RuntimeError(f"Node {source_node!s} has no missing input handler") from exc

    def __call__(self, *args, **kwargs):
        newinput = self._source_handler(*args, **kwargs)
        self._target_node.inputs.add(newinput)

        if self._inherit_outputs and newinput.child_output is not None:
            self._target_node.outputs.add(newinput.child_output)

        return newinput


InputStrateges = {
    AddNewInput,
    AddNewInputAddNewOutput,
    AddNewInputAddAndKeepSingleOutput,
    AddNewInputAddNewOutputForBlock,
    AddNewInputAddNewOutputForNInputs,
    InheritInputStrategy,
}
