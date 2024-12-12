from numpy import arccos, arcsin, arctan, cos, sin, tan

from .abstract import OneToOneNode


class Cos(OneToOneNode):
    """Cos function"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "cos")

    def _function(self):
        for indata, outdata in zip(self.inputs.iter_data(), self.outputs.iter_data_unsafe()):
            cos(indata, out=outdata)


class Sin(OneToOneNode):
    """Sin function"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "sin")

    def _function(self):
        for indata, outdata in zip(self.inputs.iter_data(), self.outputs.iter_data_unsafe()):
            sin(indata, out=outdata)


class ArcCos(OneToOneNode):
    """ArcCos function"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "acos")

    def _function(self):
        for indata, outdata in zip(self.inputs.iter_data(), self.outputs.iter_data_unsafe()):
            arccos(indata, out=outdata)


class ArcSin(OneToOneNode):
    """ArcSin function"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "asin")

    def _function(self):
        for indata, outdata in zip(self.inputs.iter_data(), self.outputs.iter_data_unsafe()):
            arcsin(indata, out=outdata)


class Tan(OneToOneNode):
    """Tan function"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "tan")

    def _function(self):
        for indata, outdata in zip(self.inputs.iter_data(), self.outputs.iter_data_unsafe()):
            tan(indata, out=outdata)


class ArcTan(OneToOneNode):
    """Arctan function"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "atan")

    def _function(self):
        for inp, out in zip(self.inputs, self.outputs):
            arctan(inp.data, out=out.data_unsafe)
