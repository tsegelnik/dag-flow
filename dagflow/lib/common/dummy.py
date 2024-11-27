from ...core.node import Node


class Dummy(Node):
    """A dummy class that does nothing"""

    __slots__ = ()

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def _typefunc(self):
        pass
