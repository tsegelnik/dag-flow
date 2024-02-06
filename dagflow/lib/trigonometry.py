from numpy import arccos
from numpy import arcsin
from numpy import arctan
from numpy import cos
from numpy import sin
from numpy import tan

from .OneToOneNode import OneToOneNode


class Cos(OneToOneNode):
    """Cos function"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "cos")

    def _fcn(self):
        for inp, out in zip(self.inputs, self.outputs):
            cos(inp.data, out=out.data)


class Sin(OneToOneNode):
    """Sin function"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "sin")

    def _fcn(self):
        for inp, out in zip(self.inputs, self.outputs):
            sin(inp.data, out=out.data)


class ArcCos(OneToOneNode):
    """ArcCos function"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "acos")

    def _fcn(self):
        for inp, out in zip(self.inputs, self.outputs):
            arccos(inp.data, out=out.data)


class ArcSin(OneToOneNode):
    """ArcSin function"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "asin")

    def _fcn(self):
        for inp, out in zip(self.inputs, self.outputs):
            arcsin(inp.data, out=out.data)


class Tan(OneToOneNode):
    """Tan function"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "tan")

    def _fcn(self):
        for inp, out in zip(self.inputs, self.outputs):
            tan(inp.data, out=out.data)


class ArcTan(OneToOneNode):
    """Arctan function"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "atan")

    def _fcn(self):
        for inp, out in zip(self.inputs, self.outputs):
            arctan(inp.data, out=out.data)
