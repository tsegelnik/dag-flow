from numpy import arctan, cos, sin, tan, arccos, arcsin

from .One2One import One2One


class Cos(One2One):
    """Cos function"""

    def _fcn(self, _, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            out.data[:] = cos(inp.data)
        return list(outputs.iter_data())


class Sin(One2One):
    """Sin function"""

    def _fcn(self, _, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            out.data[:] = sin(inp.data)
        return list(outputs.iter_data())

class ArcCos(One2One):
    """ArcCos function"""

    def _fcn(self, _, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            out.data[:] = arccos(inp.data)
        return list(outputs.iter_data())


class ArcSin(One2One):
    """ArcSin function"""

    def _fcn(self, _, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            out.data[:] = arcsin(inp.data)
        return list(outputs.iter_data())


class Tan(One2One):
    """Tan function"""

    def _fcn(self, _, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            out.data[:] = tan(inp.data)
        return list(outputs.iter_data())


class Arctan(One2One):
    """Arctan function"""

    def _fcn(self, _, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            out.data[:] = arctan(inp.data)
        return list(outputs.iter_data())
