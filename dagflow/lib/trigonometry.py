from numpy import arctan, cos, sin, tan, arccos, arcsin

from .OneToOneNode import OneToOneNode


class Cos(OneToOneNode):
    """Cos function"""

    def _fcn(self):
        for inp, out in zip(self.inputs, self.outputs):
            cos(inp.data, out=out.data)


class Sin(OneToOneNode):
    """Sin function"""

    def _fcn(self):
        for inp, out in zip(self.inputs, self.outputs):
            sin(inp.data, out=out.data)


class ArcCos(OneToOneNode):
    """ArcCos function"""

    def _fcn(self):
        for inp, out in zip(self.inputs, self.outputs):
            arccos(inp.data, out=out.data)


class ArcSin(OneToOneNode):
    """ArcSin function"""

    def _fcn(self):
        for inp, out in zip(self.inputs, self.outputs):
            arcsin(inp.data, out=out.data)


class Tan(OneToOneNode):
    """Tan function"""

    def _fcn(self):
        for inp, out in zip(self.inputs, self.outputs):
            tan(inp.data, out=out.data)


class ArcTan(OneToOneNode):
    """Arctan function"""

    def _fcn(self):
        for inp, out in zip(self.inputs, self.outputs):
            arctan(inp.data, out=out.data)
