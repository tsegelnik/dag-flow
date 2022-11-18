

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

    input_fmt = "input_{:02d}"

    def __init__(self, node=None, fmt=None):
        super().__init__(node)
        if fmt is not None:
            self.input_fmt = fmt

    def __call__(self, idx=None, scope=None, **kwargs):
        inp = self.node._add_input(
            self.input_fmt.format(
                idx if idx is not None else len(self.node.inputs)
            ),
            **kwargs,
        )
        if self.node.closed:
            inp.close()
        return inp


class MissingInputAddPair(MissingInputAdd):
    """
    Adds an input for each output in >> operator.
    Adds an output for each new input
    """

    output_fmt = "output_{:02d}"

    def __init__(self, node=None, input_fmt=None, output_fmt=None):
        super().__init__(node, input_fmt)
        if output_fmt is not None:
            self.output_fmt = output_fmt

    def __call__(self, idx=None, scope=None):
        idx_out = len(self.node.outputs)
        out = self.node._add_output(self.output_fmt.format(idx_out))
        return super().__call__(idx, child_output=out, scope=scope)


class MissingInputAddOne(MissingInputAdd):
    """
    Adds an input for each output in >> operator.
    Adds only one output if needed
    """

    output_fmt = "output_{:02d}"
    add_child_output = False

    def __init__(
        self,
        node=None,
        input_fmt=None,
        output_fmt=None,
        add_child_output=False,
    ):
        super().__init__(node, input_fmt)
        if output_fmt is not None:
            self.output_fmt = output_fmt
        self.add_child_output = add_child_output

    def __call__(self, idx=None, scope=None):
        if (idx_out := len(self.node.outputs)) == 0:
            out = self.node._add_output(self.output_fmt.format(idx_out))
            if self.node.closed:
                out.close()
        else:
            out = self.node.outputs[-1]
        if self.add_child_output:
            return super().__call__(idx, child_output=out, scope=scope)
        return super().__call__(idx, scope=scope)


class MissingInputAddEach(MissingInputAdd):
    """
    Adds an output for each block (for each >> operation)
    """

    output_fmt = "output_{:02d}"
    add_child_output = False
    scope = 0

    def __init__(
        self,
        node=None,
        input_fmt=None,
        output_fmt=None,
        add_child_output=False,
    ):
        super().__init__(node, input_fmt)
        if output_fmt is not None:
            self.output_fmt = output_fmt
        self.add_child_output = add_child_output

    def __call__(self, idx=None, scope=None):
        if scope == self.scope:
            out = self.node.outputs[-1]
        else:
            out = self.node._add_output(
                self.output_fmt.format(len(self.node.outputs))
            )
            self.scope = scope
        return (
            super().__call__(idx, child_output=out, scope=scope)
            if self.add_child_output
            else super().__call__(idx, scope=scope)
        )
