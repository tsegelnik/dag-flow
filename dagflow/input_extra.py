from typing import Optional, Union

class SimpleFormatter():
    _base: str
    _numfmt: str
    def __init__(self, base: str, numfmt: str = '_{:02d}'):
        self._base = base
        self._numfmt = numfmt

    @staticmethod
    def from_string(string: str):
        if '{' in string:
             return string

        return SimpleFormatter(string)

    def format(self, num: int) -> str:
        if num>0:
            return self._base+self._numfmt.format(num)

        return self._base


class MissingInputHandler:
    """
    Handler to implement behaviour when output
    is connected to the missing input with >>/<<
    """

    _node = None

    def __init__(self, node=None):
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

    def __init__(self, node=None):
        super().__init__(node)

    def __call__(self, idx=None, scope=None):
        raise RuntimeError(
            "Unable to iterate inputs further. "
            "No additional inputs may be created"
        )


class MissingInputAdd(MissingInputHandler):
    """Adds an input for each output in >> operator"""

    input_fmt: Union[str,SimpleFormatter] = SimpleFormatter("input", "_{:02d}")
    input_kws: dict
    output_fmt: Union[str,SimpleFormatter] = SimpleFormatter("output", "_{:02d}")
    output_kws: dict

    def __init__(
        self,
        node=None,
        *,
        input_fmt: Optional[Union[str,SimpleFormatter]] = None,
        input_kws: Optional[dict] = None,
        output_fmt: Optional[Union[str,SimpleFormatter]] = None,
        output_kws: Optional[dict] = None,
    ):
        if input_kws is None:
            input_kws = {}
        if output_kws is None:
            output_kws = {}
        super().__init__(node)
        self.input_kws = input_kws
        self.output_kws = output_kws
        if input_fmt is not None:
            self.input_fmt = SimpleFormatter.from_string(input_fmt)
        if output_fmt is not None:
            self.output_fmt = SimpleFormatter.from_string(output_fmt)

    def __call__(self, idx=None, scope=None, **kwargs):
        kwargs_combined = dict(self.input_kws, **kwargs)
        return self.node._add_input(
            self.input_fmt.format(
                idx if idx is not None else len(self.node.inputs)
            ),
            **kwargs_combined,
        )


class MissingInputAddPair(MissingInputAdd):
    """
    Adds an input for each output in >> operator.
    Adds an output for each new input
    """

    def __init__(self, node=None, **kwargs):
        super().__init__(node, **kwargs)

    def __call__(self, idx=None, scope=None):
        idx_out = len(self.node.outputs)
        out = self.node._add_output(
            self.output_fmt.format(idx_out), **self.output_kws
        )
        return super().__call__(idx, child_output=out, scope=scope)


class MissingInputAddOne(MissingInputAdd):
    """
    Adds an input for each output in >> operator.
    Adds only one output if needed
    """

    add_child_output = False

    def __init__(self, node=None, *, add_child_output: bool = False, **kwargs):
        super().__init__(node, **kwargs)
        self.add_child_output = add_child_output

    def __call__(self, idx=None, scope=None):
        if (idx_out := len(self.node.outputs)) == 0:
            out = self.node._add_output(
                self.output_fmt.format(idx_out), **self.output_kws
            )
        else:
            out = self.node.outputs[-1]
        if self.add_child_output:
            return super().__call__(idx, child_output=out, scope=scope)
        return super().__call__(idx, scope=scope)


class MissingInputAddEach(MissingInputAdd):
    """
    Adds an output for each block (for each >> operation)
    """

    add_child_output = False
    scope = 0

    def __init__(self, node=None, *, add_child_output=False, **kwargs):
        super().__init__(node, **kwargs)
        self.add_child_output = add_child_output

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
