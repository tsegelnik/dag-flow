import cython
import functools
from libc.stdlib cimport malloc, free

from node cimport Node, CNode
from functions cimport fcn_sum, fcn_product, fcn_sin


cdef class Input(Node):
    cdef double* data
    cdef int size

    def __init__(self, data):
        super().__init__(inputs=None)
        self.size = len(data)

        self.data = <double *>malloc(sizeof(double) * self.size)
        for i in range(self.size):
            self.data[i] = data[i]

    cdef void _setup_cnode(self):
        self.cnode.inputs = cython.NULL
        self.cnode.input_sizes = cython.NULL
        self.cnode.input_count = 0
        self.cnode.data = self.data

    @functools.cache
    def get_size(self):
        return self.size

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