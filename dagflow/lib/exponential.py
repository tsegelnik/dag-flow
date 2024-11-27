from numpy import exp, expm1, log, log1p, log10

from .abstract import OneToOneNode


class Exp(OneToOneNode):
    """exp(x) function"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "exp")

    def _function(self):
        for inp, out in zip(self.inputs, self.outputs):
            exp(inp.data, out=out.data)


class Expm1(OneToOneNode):
    """exp(x)-1 function"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "exp-1")

    def _function(self):
        for inp, out in zip(self.inputs, self.outputs):
            expm1(inp.data, out=out.data)


class Log(OneToOneNode):
    """log(x) function"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "log")

    def _function(self):
        for inp, out in zip(self.inputs, self.outputs):
            log(inp.data, out=out.data)


class Log1p(OneToOneNode):
    """log(x+1) function"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "log(x+1)")

    def _function(self):
        for inp, out in zip(self.inputs, self.outputs):
            log1p(inp.data, out=out.data)


class Log10(OneToOneNode):
    """log10(x) function"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "log₁₀")

    def _function(self):
        for inp, out in zip(self.inputs, self.outputs):
            log10(inp.data, out=out.data)
