from typing import Callable

from numpy import add, copyto, divide, multiply, sqrt, square
from numpy.typing import NDArray

from .ManyToOneNode import ManyToOneNode
from .OneToOneNode import OneToOneNode


class Sum(ManyToOneNode):
    """Sum of all the inputs together"""

    __slots__ = (
        "_input_node_callbacks",
        "_input_data0",
        "_input_data",
        "_output_data",
    )

    _input_node_callbacks: list[Callable]
    _input_data0: NDArray
    _input_data: list[NDArray]
    _output_data: NDArray

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("broadcastable", True)
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "Σ")

    def _fcn(self):
        for callback in self._input_node_callbacks:
            callback()

        copyto(self._output_data, self._input_data0)
        for input_data in self._input_data:
            add(self._output_data, input_data, out=self._output_data)

    def _post_allocate(self):
        self._input_node_callbacks = []
        self._input_data = []
        for input in self.inputs:
            node = input.parent_node
            if not node in self._input_node_callbacks:
                self._input_node_callbacks.append(node.touch)

            self._input_data.append(input.data_unsafe)

        self._input_data0, self._input_data = self._input_data[0], self._input_data[1:]
        self._output_data = self.outputs["result"].data_unsafe


class Product(ManyToOneNode):
    """Product of all the inputs together"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("broadcastable", True)
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "Π")

    def _fcn(self):
        out = self.outputs["result"].data
        copyto(out, self.inputs[0].data)
        for _input in self.inputs[1:]:
            multiply(out, _input.data, out=out)


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

    def _fcn(self):
        out = self.outputs[0].data
        copyto(out, self.inputs[0].data.copy())
        for _input in self.inputs[1:]:
            divide(out, _input.data, out=out)


class Square(OneToOneNode):
    """Square function"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "x²")

    def _fcn(self):
        for inp, out in zip(self.inputs, self.outputs):
            square(inp.data, out=out.data)


class Sqrt(OneToOneNode):
    """Square function"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "√x")

    def _fcn(self):
        for inp, out in zip(self.inputs, self.outputs):
            sqrt(inp.data, out=out.data)
