import cython
import functools
from libc.stdlib cimport malloc, free
cimport numpy as cnp
import numpy as np

from node cimport Node, CNode
from functions cimport fcn_integration, fcn_sum, fcn_product, fcn_sin, fcn_cosh


cdef class Input(Node):
    cdef double* data
    cdef int size

    def __init__(self, data):
        super().__init__(inputs=None)
        self.size = len(data)

        self.data = <double *>malloc(sizeof(double) * self.size)
        for i in range(self.size):
            self.data[i] = data[i]

    cdef CNode* _compile(self):
        cdef double* data_ptr = &self.data[0]
        self.cnode.input_sizes = <int *>malloc(sizeof(int))
        
        self.cnode.inputs = cython.NULL
        self.cnode.input_sizes[0] = self.size
        self.cnode.input_count = 1
        self.cnode.fcn = self.fcn
        self.cnode.data = data_ptr
        self.cnode.data_size = self.size

        return self.cnode

    @functools.cache
    def get_size(self):
        return self.size

    def __dealloc__(self):
        if self.cnode is not NULL:
            free(self.cnode)


cdef class Integrator(Node):
    def __init__(self, inputs=None):
        super().__init__(inputs)
        self.fcn = fcn_integration

    @functools.cache
    def get_size(self):
        return 1


cdef class Sum(Node):
    def __init__(self, inputs=None):
        super().__init__(inputs)
        self.fcn = fcn_sum

    @functools.cache
    def get_size(self):
        return self.inputs[0].get_size() if self.inputs else 0


cdef class Product(Node):
    def __init__(self, inputs=None):
        super().__init__(inputs)
        self.fcn = fcn_product

    @functools.cache
    def get_size(self):
        return self.inputs[0].get_size() if self.inputs else 0


cdef class Sin(Node):
    def __init__(self, inputs=None):
        super().__init__(inputs)
        self.fcn = fcn_sin

    @functools.cache
    def get_size(self):
        return self.inputs[0].get_size() if self.inputs else 0


cdef class Cosh(Node):
    def __init__(self, inputs=None):
        super().__init__(inputs)
        self.fcn = fcn_cosh

    @functools.cache
    def get_size(self):
        return self.inputs[0].get_size() if self.inputs else 0
