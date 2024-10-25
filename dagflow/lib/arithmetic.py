from numpy import add, copyto, divide, multiply, sqrt, square

from .abstract import ManyToOneNode
from .abstract import OneToOneNode


class Sum(ManyToOneNode):
    """Sum of all the inputs together"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("broadcastable", True)
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "Σ")

    def _function(self):
        for callback in self._input_nodes_callbacks:
            callback()

        output_data = self._output_data
        copyto(output_data, self._input_data0)
        for input_data in self._input_data_other:
            add(output_data, input_data, out=output_data)

class Product(ManyToOneNode):
    """Product of all the inputs together"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("broadcastable", True)
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "Π")

    def _function(self):
        for callback in self._input_nodes_callbacks:
            callback()

        output_data = self._output_data
        copyto(output_data, self._input_data0)
        for _input_data in self._input_data_other:
            multiply(output_data, _input_data, out=output_data)


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

    def _function(self):
        for callback in self._input_nodes_callbacks:
            callback()

        output_data = self._output_data
        copyto(output_data, self._input_data0)
        for _input_data in self._input_data_other:
            divide(self._output_data, _input_data, out=self._output_data)


class Square(OneToOneNode):
    """Square function"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "x²")

    def _function(self):
        for callback in self._input_nodes_callbacks:
            callback()

        for input_data, output_data in self._input_output_data:
            square(input_data, out=output_data)


class Sqrt(OneToOneNode):
    """Square function"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "√x")

    def _function(self):
        for callback in self._input_nodes_callbacks:
            callback()

        for input_data, output_data in self._input_output_data:
            sqrt(input_data, out=output_data)
