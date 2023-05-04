from ..nodes import FunctionNode


class Dummy(FunctionNode):
    """A dummy class that does nothing"""

    def __init__(self, name, **kwargs):
        kwargs.setdefault("fcn", lambda: None)
        kwargs.setdefault("typefunc", lambda: None)
        super().__init__(name, **kwargs)
