import functools
from ctypes import POINTER, c_double, cast

from synthetic_graph.graph_ctypes.node import Node, CNode, library, CFcnType


class Input(Node):
    __slots__ = (
        "data"
    )

    data: list[float]

    def __init__(self, data: list[float]):
        super().__init__(inputs=None)
        self.data = data

    def to_c_struct(self):
        data_array = (c_double * len(self.data))(*self.data)
        data_ptr = cast(data_array, POINTER(c_double))
        self.cnode = CNode(
            inputs=None,
            input_sizes=None,
            input_count=0,
            fcn=self.get_fcn(),
            data=data_ptr
        )
        return self.cnode

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
