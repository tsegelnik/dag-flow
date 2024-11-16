import functools
from abc import abstractmethod, ABC
from ctypes import POINTER, c_int, c_double, CFUNCTYPE, Structure, pointer, CDLL, cast

library = CDLL('./synthetic_graph/graph_ctypes/source/libnode.so')


class CNode(Structure):
    pass


CFcnType = CFUNCTYPE(POINTER(c_double), POINTER(POINTER(c_double)), POINTER(c_int), c_int)
CNode._fields_ = [
    ("inputs", POINTER(POINTER(CNode))),
    ("input_sizes", POINTER(c_int)),
    ("input_count", c_int),
    ("fcn", CFcnType),
    ("data", POINTER(c_double))
]

run_node = library.run_node
run_node.restype = POINTER(c_double)
run_node.argtypes = [POINTER(CNode)]


class Node(ABC):
    __slots__ = (
        "inputs",
        "cnode",
    )

    inputs: list["Node"]
    cnode: CNode | None

    def __init__(
            self,
            inputs: list["Node"] | None = None,
    ):
        if inputs is not None:
            self.inputs = inputs
        else:
            self.inputs = []

    def __rshift__(self, other):
        other.inputs.append(self)
        return other

    def compile(self):
        input_nodes = (POINTER(CNode) * len(self.inputs))()
        input_sizes = (c_int * len(self.inputs))()

        for i, input_node in enumerate(self.inputs):
            input_nodes[i] = pointer(input_node.compile())
            input_sizes[i] = input_node.get_size()

        data_array = (c_double * self.get_size())()
        data_ptr = cast(data_array, POINTER(c_double))

        self.cnode = CNode(
            inputs=input_nodes,
            input_sizes=input_sizes,
            input_count=len(self.inputs),
            fcn=self.get_fcn(),
            data=data_ptr,
        )
        return self.cnode

    def run(self):
        return run_node(pointer(self.cnode))

    @functools.cache
    def get_size(self):
        return self.inputs[0].get_size()

    @staticmethod
    @abstractmethod
    def get_fcn():
        pass

