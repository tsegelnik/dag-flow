import cython
import functools
from libc.stdlib cimport malloc, free

from node cimport Node
from functions cimport fcn_sum, fcn_product, fcn_sin


cdef class Input(Node):
    cdef double* data
    cdef int data_size

    def __init__(self, data):
        super().__init__(inputs=None)
        self.data_size = len(data)

        self.data = <double *>malloc(sizeof(double) * self.data_size)
        for i in range(self.data_size):
            self.data[i] = data[i]

    cdef void _setup_cnode(self):
        self.cnode.inputs = cython.NULL
        self.cnode.input_sizes = cython.NULL
        self.cnode.input_count = 0
        self.cnode.data = self.data
        self.cnode.data_size = self.data_size

    def __dealloc__(self):
        free(self.data)


cdef class Sum(Node):
    def __init__(self, inputs=None):
        super().__init__(inputs)
        self.fcn = fcn_sum


cdef class Product(Node):
    def __init__(self, inputs=None):
        super().__init__(inputs)
        self.fcn = fcn_product


cdef class Sin(Node):
    def __init__(self, inputs=None):
        super().__init__(inputs)
        self.fcn = fcn_sin