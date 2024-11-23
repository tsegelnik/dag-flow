import numpy as np
import numpy.typing as npt
from typing import NewType

from synthetic_graph.graph_python.node import Node

Data = NewType("Data", npt.ArrayLike)


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
            [input_node.run() for input_node in self.inputs],
            axis=0
        )


class Product(Node):
    def fcn(self):
        return np.prod(
            [input_node.run() for input_node in self.inputs],
            axis=0
        )


class Sin(Node):
    def fcn(self):
        return np.sin(self.inputs[0].run())
 