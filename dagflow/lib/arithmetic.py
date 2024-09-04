from numpy import add, copyto, divide, multiply, sqrt, square

from .ManyToOneNode import ManyToOneNode
from .OneToOneNode import OneToOneNode


class Sum(ManyToOneNode):
    """Sum of all the inputs together"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("broadcastable", True)
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "Σ")

    def _fcn(self):
        out = self.outputs["result"].data
        copyto(out, self.inputs[0].data)
        for _input in self.inputs[1:]:
            add(out, _input.data, out=out)


class Product(ManyToOneNode):
    """Product of all the inputs together"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("broadcastable", True)
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "Π")

    def _fcn(self):
        out = self.outputs["result"].data
        copyto(out, self.inputs[0].data)
        for _input in self.inputs[1:]:
            multiply(out, _input.data, out=out)


class Division(ManyToOneNode):
    """
    Division of the first input to other

    .. note:: a division by zero returns `nan`
    """

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("broadcastable", True)
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "÷")

    def _fcn(self):
        out = self.outputs[0].data
        copyto(out, self.inputs[0].data.copy())
        for _input in self.inputs[1:]:
            divide(out, _input.data, out=out)


class Square(OneToOneNode):
    """Square function"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "x²")

    def _fcn(self):
        for inp, out in zip(self.inputs, self.outputs):
            square(inp.data, out=out.data)


class Sqrt(OneToOneNode):
    """Square function"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "√x")

    def _fcn(self):
        for inp, out in zip(self.inputs, self.outputs):
            sqrt(inp.data, out=out.data)
