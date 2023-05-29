from ..nodes import FunctionNode


class Dummy(FunctionNode):
    """A dummy class that does nothing"""

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def _typefunc(self):
        pass
