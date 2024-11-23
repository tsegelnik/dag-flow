import functools
from ctypes import POINTER, c_double, cast

from synthetic_graph.graph_python_ctypes.node import Node, library, CFcnType


class Input(Node):
    __slots__ = (
        "data"
    )

    data: list[float]

    def __init__(self, data: list[float]):
        self.data = data
        super().__init__(inputs=None)

    def compile(self):
        data_array = (c_double * len(self.data))(*self.data)
        self.c_data = cast(data_array, POINTER(c_double))

    @functools.cache
    def get_size(self):
        return len(self.data)

    @staticmethod
    def get_fcn():
        return CFcnType(("input_fcn", library))


class Sum(Node):
    @staticmethod
    def get_fcn():
        return CFcnType(("sum_fcn", library))


class Product(Node):
    @staticmethod
    def get_fcn():
        return CFcnType(("product_fcn", library))


class Sin(Node):
    @staticmethod
    def get_fcn():
        return CFcnType(("sin_fcn", library))
