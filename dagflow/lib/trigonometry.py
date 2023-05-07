from numpy import arctan, cos, sin, tan, arccos, arcsin

from .OneToOneNode import OneToOneNode


class Cos(OneToOneNode):
    """Cos function"""

    def _fcn(self, _, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            cos(inp.data, out=out.data)
        return list(outputs.iter_data())


class Sin(OneToOneNode):
    """Sin function"""

    def _fcn(self, _, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            sin(inp.data, out=out.data)
        return list(outputs.iter_data())


class ArcCos(OneToOneNode):
    """ArcCos function"""

    def _fcn(self, _, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            arccos(inp.data, out=out.data)
        return list(outputs.iter_data())


class ArcSin(OneToOneNode):
    """ArcSin function"""

    def _fcn(self, _, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            arcsin(inp.data, out=out.data)
        return list(outputs.iter_data())


class Tan(OneToOneNode):
    """Tan function"""

    def _fcn(self, _, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            tan(inp.data, out=out.data)
        return list(outputs.iter_data())


class ArcTan(OneToOneNode):
    """Arctan function"""

    def _fcn(self, _, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            arctan(inp.data, out=out.data)
        return list(outputs.iter_data())
