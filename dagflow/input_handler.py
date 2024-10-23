from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from .exception import InitializationError

if TYPE_CHECKING:
    from .node import Node


class Formatter:
    __slots__ = ()

    def format(self, num: int) -> str:
        raise RuntimeError("Virtual method called")

    @staticmethod
    def from_string(string: str):
        return string if "{" in string else SimpleFormatter(string)

    @staticmethod
    def from_sequence(seq: Sequence[str]):
        return SequentialFormatter(seq)

    @staticmethod
    def from_value(value: str | Sequence[str] | Formatter):
        if isinstance(value, Formatter):
            return value
        elif isinstance(value, str):
            return Formatter.from_string(value)
        elif isinstance(value, Sequence):
            return Formatter.from_sequence(value)

        raise InitializationError(
            f"Expect str, Tuple[str] or Formatter, got {type(value).__name__}"
        )


Formattable = Formatter | str


class SimpleFormatter(Formatter):
    __slots__ = ("_base", "_numfmt")
    _base: str
    _numfmt: str

    def __init__(self, base: str, numfmt: str = "_{:02d}"):
        self._base = base
        self._numfmt = numfmt

    def format(self, num: int) -> str:
        return self._base + self._numfmt.format(num) if num > 0 else self._base


class SequentialFormatter(Formatter):
    __slots__ = ("_base", "_numfmt", "_startidx")

    _base: tuple[str, ...]
    _numfmt: str
    _startidx: int

    def __init__(self, base: Sequence[str], numfmt: str = "_{:02d}", startidx: int = 0):
        self._base = tuple(base)
        self._numfmt = numfmt
        self._startidx = startidx

    def format(self, num: int) -> str:
        num -= self._startidx
        idx = num % len(self._base)
        groupnum = num // len(self._base)
        base = self._base[idx]
        if groupnum > 0:
            return base + self._numfmt.format(groupnum)
        elif num < 0:
            raise ValueError(f"SequentialFormatter got num={num}<0")

        return base


class MissingInputHandler:
    """
    Handler to implement behaviour when output
    is connected to the missing input with >>/<<
    """

    __slots__ = ("_node",)

    _node: Node | None

    def __init__(self, node: Node | None = None):
        self.node = node

    @property
    def node(self):
        return self._node

    @node.setter
    def node(self, node):
        self._node = node

    def __call__(self, idx=None, scope=None):
        pass


class MissingInputFail(MissingInputHandler):
    """Default missing input handler: issues and exception"""

    __slots__ = ()

    def __init__(self, node=None):
        super().__init__(node)

    def __call__(self, idx=None, scope=None):
        raise RuntimeError("Unable to iterate inputs further. No additional inputs may be created")


class MissingInputAdd(MissingInputHandler):
    """Adds an input for each output in >> operator"""

    __slots__ = ("input_fmt", "input_kws", "output_fmt", "output_kws")

    input_fmt: Formattable
    input_kws: dict
    output_fmt: Formattable
    output_kws: dict

    def __init__(
        self,
        node: Node | None = None,
        *,
        input_fmt: str | Sequence[str] | Formatter = SimpleFormatter("input", "_{:02d}"),
        input_kws: dict | None = None,
        output_fmt: str | Sequence[str] | Formatter = SimpleFormatter("output", "_{:02d}"),
        output_kws: dict | None = None,
    ):
        if input_kws is None:
            input_kws = {}
        if output_kws is None:
            output_kws = {}
        super().__init__(node)
        self.input_kws = input_kws
        self.output_kws = output_kws
        self.input_fmt = Formatter.from_value(input_fmt)
        self.output_fmt = Formatter.from_value(output_fmt)

    def __call__(self, idx=None, scope=None, *, fmt: Formattable | None = None, **kwargs):
        kwargs_combined = dict(self.input_kws, **kwargs)
        if fmt is None:
            fmt = self.input_fmt
        return self.node._add_input(
            fmt.format(idx if idx is not None else len(self.node.inputs)),
            **kwargs_combined,
        )


class MissingInputAddPair(MissingInputAdd):
    """
    Adds an input for each output in >> operator.
    Adds an output for each new input
    """

    __slots__ = ()

    def __init__(self, node=None, **kwargs):
        super().__init__(node, **kwargs)

    def __call__(self, idx=None, scope=None, idx_out=None):
        if idx_out is None:
            idx_out = len(self.node.outputs)
        out = self.node._add_output(self.output_fmt.format(idx_out), **self.output_kws)
        return super().__call__(idx, child_output=out, scope=scope)


class MissingInputAddOne(MissingInputAdd):
    """
    Adds an input for each output in >> operator.
    Adds only one output if needed
    """

    __slots__ = ("add_child_output",)

    add_child_output: bool

    def __init__(self, node=None, *, add_child_output: bool = False, **kwargs):
        super().__init__(node, **kwargs)
        self.add_child_output = add_child_output

    def __call__(self, idx=None, scope=None, idx_out=None):
        if idx_out is not None:
            out = self.node._add_output(self.output_fmt.format(idx_out), **self.output_kws)
        elif (idx_out := len(self.node.outputs)) == 0:
            out = self.node._add_output(self.output_fmt.format(idx_out), **self.output_kws)
        else:
            out = self.node.outputs[-1]
        if self.add_child_output:
            return super().__call__(idx, child_output=out, scope=scope)
        return super().__call__(idx, scope=scope)


class MissingInputAddEach(MissingInputAdd):
    """
    Adds an output for each block (for each >> operation)
    """

    __slots__ = ("add_child_output", "scope")

    add_child_output: bool
    scope: int

    def __init__(self, node=None, *, add_child_output=False, **kwargs):
        super().__init__(node, **kwargs)
        self.add_child_output = add_child_output
        self.scope = 0

    def __call__(self, idx=None, scope=None):
        if scope == self.scope != 0:
            out = self.node.outputs[-1]
        else:
            out = self.node._add_output(
                self.output_fmt.format(len(self.node.outputs)),
                **self.output_kws,
            )
            self.scope = scope
        if self.add_child_output:
            return super().__call__(idx, child_output=out, scope=scope)
        return super().__call__(idx, scope=scope)


class MissingInputAddEachN(MissingInputAdd):
    """
    Adds an output for each N inputs
    """

    __slots__ = ("add_child_output", "scope", "n", "input_fmts")

    add_child_output: bool
    scope: int
    n: int

    def __init__(
        self,
        n: int,
        node=None,
        *,
        init_with_no_inputs: bool = False,
        add_child_output=False,
        **kwargs,
    ):
        super().__init__(node, **kwargs)
        self.n = n
        self.scope = 1 if init_with_no_inputs else 0
        self.add_child_output = add_child_output

    def __call__(self, idx=None, scope=None):
        if scope == self.scope != 0:
            out = self.node.outputs[-1]
        elif self.node.inputs.len_pos() % self.n == 0:
            out = self.node._add_output(
                self.output_fmt.format(len(self.node.outputs)),
                **self.output_kws,
            )
            self.scope = scope
        if self.add_child_output:
            return super().__call__(idx, child_output=out, scope=scope)
        return super().__call__(idx, scope=scope)
