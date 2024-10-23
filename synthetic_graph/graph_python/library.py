import numpy as np
import numpy.typing as npt
from typing import NewType

from synthetic_graph.graph_python.node import Node

Data = NewType('Data', npt.ArrayLike)


class Input(Node):
    __slots__ = (
        "data"
    )

    def __init__(self, data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = data

    def fcn(self):
        return self.data


class Sum(Node):
    def fcn(self):
        return np.sum(
            [input_node() for input_node in self.inputs],
            axis=0
        )


class Product(Node):
    def fcn(self):
        return np.prod(
            [input_node() for input_node in self.inputs],
            axis=0
        )


class Integrator(Node):
    def fcn(self):
        values = self.inputs[0]()
        bins = self.inputs[1]()
        return np.trapz(values, bins)


class Repeater(Node):
    __slots__ = (
        "count"
    )

    def __init__(self, count, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count = count

    def fcn(self):
        data = self.inputs[0]()
        return np.tile(data, self.count)


class Sin(Node):
    def fcn(self):
        data = np.clip(self.inputs[0](), 0, 1)
        return np.sin(data)


class Cosh(Node):
    def fcn(self):
        data = np.clip(self.inputs[0](), 0, 1)
        return np.cosh(data)


class Sinh(Node):
    def fcn(self):
        data = np.clip(self.inputs[0](), 0, 1)
        return np.sinh(data)


class Tan(Node):
    def fcn(self):
        data = np.clip(self.inputs[0](), 0, 1)
        return np.tan(data)
