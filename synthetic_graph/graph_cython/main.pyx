import cython
import numpy as np
cimport numpy as cnp


ctypedef cnp.float64_t DTYPE_t

cdef class Node:
    cdef list fcn_inputs

    def __init__(self, inputs=None):
        if inputs is not None:
            self.fcn_inputs = [inp.get_fcn() for inp in inputs]
        else:
            self.fcn_inputs = []

    def __call__(self):
        return self.fcn()

    def __rshift__(self, other: "Node"):
        other.fcn_inputs.append(self.get_fcn())
        return other

    def get_fcn(self):
        return lambda: self.fcn()

    cdef fcn(self):
        pass


cdef class Input(Node):
    cdef cnp.ndarray data

    def __init__(self, data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = data

    cdef fcn(self):
        return self.data


cdef class Repeater(Node):
    cdef int count

    def __init__(self, count, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count = count

    cdef fcn(self):
        data = self.fcn_inputs[0]()
        return np.tile(data, self.count)


cdef class Sum(Node):
    cdef fcn(self):
        return np.sum(
            [fcn_input() for fcn_input in self.fcn_inputs],
            axis=0
        )


cdef class Product(Node):
    cdef fcn(self):
        return np.prod(
            [fcn_input() for fcn_input in self.fcn_inputs],
            axis=0
        )


cdef class Integrator(Node):
    cdef fcn(self):
        values = self.fcn_inputs[0]()
        bins = self.fcn_inputs[1]()
        return np.trapz(values, bins)


cdef class Sin(Node):
    cdef fcn(self):
        data = np.clip(self.fcn_inputs[0](), 0, 1)
        return np.sin(data)


cdef class Cosh(Node):
    cdef fcn(self):
        data = np.clip(self.fcn_inputs[0](), 0, 1)
        return np.cosh(data)


cdef class Sinh(Node):
    cdef fcn(self):
        data = np.clip(self.fcn_inputs[0](), 0, 1)
        return np.sinh(data)


cdef class Tan(Node):
    cdef fcn(self):
        data = np.clip(self.fcn_inputs[0](), 0, 1)
        return np.tan(data)
