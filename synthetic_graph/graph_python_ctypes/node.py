import functools
from abc import abstractmethod, ABC
from ctypes import POINTER, c_int, c_double, CDLL, cast, CFUNCTYPE

library = CDLL('./graph_python_ctypes/source/libnode.so')

CFcnType = CFUNCTYPE(POINTER(c_double), POINTER(POINTER(c_double)), POINTER(c_int), c_int)


class Node(ABC):
    __slots__ = (
        "inputs",
        "c_inputs",
        "c_input_sizes",
        "c_data",
    )

    inputs: list["Node"]
    c_inputs: POINTER(POINTER(c_double))
    c_input_sizes: POINTER(c_int)
    c_data: POINTER(c_double)

    def __init__(
            self,
            inputs: list["Node"] | None = None,
    ):
        if inputs is not None:
            self.inputs = inputs
        else:
            self.inputs = []

        self.c_input_sizes = None
        self.c_data = None
        self.c_inputs = None

    def __rshift__(self, other):
        other.inputs.append(self)
        return other

    def compile(self):
        data_array = (c_double * self.get_size())()
        self.c_data = cast(data_array, POINTER(c_double))

        self.c_inputs = (POINTER(c_double) * len(self.inputs))()

        input_sizes = (c_int * len(self.inputs))()

        for i, input_node in enumerate(self.inputs):
            input_node.compile()
            input_sizes[i] = input_node.get_size()

        self.c_input_sizes = cast(input_sizes, POINTER(c_int))

    def run(self):
        for (i, inp) in enumerate(self.inputs):
            self.c_inputs[i] = inp.run()

        fcn = self.get_fcn()

        return fcn(
            self.c_inputs,
            self.c_input_sizes,
            len(self.inputs),
            self.c_data,
        )

    @functools.cache
    def get_size(self):
        return self.inputs[0].get_size()

    @staticmethod
    @abstractmethod
    def get_fcn():
        pass
