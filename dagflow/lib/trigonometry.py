from numpy import arctan, cos, sin, tan, arccos, arcsin

from .NodeOneToOne import NodeOneToOne


class Cos(NodeOneToOne):
    """Cos function"""

    def _fcn(self, _, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            out.data[:] = cos(inp.data)
        return list(outputs.iter_data())


class Sin(NodeOneToOne):
    """Sin function"""

    def _fcn(self, _, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            out.data[:] = sin(inp.data)
        return list(outputs.iter_data())

class ArcCos(NodeOneToOne):
    """ArcCos function"""

    def _fcn(self, _, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            out.data[:] = arccos(inp.data)
        return list(outputs.iter_data())


class ArcSin(NodeOneToOne):
    """ArcSin function"""

    def _fcn(self, _, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            out.data[:] = arcsin(inp.data)
        return list(outputs.iter_data())


class Tan(NodeOneToOne):
    """Tan function"""

    def _fcn(self, _, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            out.data[:] = tan(inp.data)
        return list(outputs.iter_data())


class Arctan(NodeOneToOne):
    """Arctan function"""

    def _fcn(self, _, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            out.data[:] = arctan(inp.data)
        return list(outputs.iter_data())
