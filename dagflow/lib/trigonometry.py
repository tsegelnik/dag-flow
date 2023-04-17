from numpy import arctan, cos, sin, tan, arccos, arcsin

from .NodeOneToOne import NodeOneToOne


class Cos(NodeOneToOne):
    """Cos function"""

    def _fcn(self, _, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            cos(inp.data, out=out.data)
        return list(outputs.iter_data())


class Sin(NodeOneToOne):
    """Sin function"""

    def _fcn(self, _, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            sin(inp.data, out=out.data)
        return list(outputs.iter_data())

class ArcCos(NodeOneToOne):
    """ArcCos function"""

    def _fcn(self, _, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            arccos(inp.data, out=out.data)
        return list(outputs.iter_data())


class ArcSin(NodeOneToOne):
    """ArcSin function"""

    def _fcn(self, _, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            arcsin(inp.data, out=out.data)
        return list(outputs.iter_data())


class Tan(NodeOneToOne):
    """Tan function"""

    def _fcn(self, _, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            tan(inp.data, out=out.data)
        return list(outputs.iter_data())


class Arctan(NodeOneToOne):
    """Arctan function"""

    def _fcn(self, _, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            arctan(inp.data, out=out.data)
        return list(outputs.iter_data())
