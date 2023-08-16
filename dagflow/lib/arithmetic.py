from numpy import add, copyto, divide, multiply

from .ManyToOneNode import ManyToOneNode


class Sum(ManyToOneNode):
    """Sum of all the inputs together"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "Σ")

    def _fcn(self):
        out = self.outputs["result"].data
        copyto(out, self.inputs[0].data)
        for _input in self.inputs[1:]:
            add(out, _input.data, out=out)
        return out


class Product(ManyToOneNode):
    """Product of all the inputs together"""

    def _fcn(self):
        out = self.outputs["result"].data
        copyto(out, self.inputs[0].data)
        for _input in self.inputs[1:]:
            multiply(out, _input.data, out=out)
        return out


class Division(ManyToOneNode):
    """
    Division of the first input to other

    .. note:: a division by zero returns `nan`
    """

    def _fcn(self):
        out = self.outputs[0].data
        copyto(out, self.inputs[0].data.copy())
        for _input in self.inputs[1:]:
            divide(out, _input.data, out=out)
        return out
