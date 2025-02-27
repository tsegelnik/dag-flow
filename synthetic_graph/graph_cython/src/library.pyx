import cython
from libc.stdlib cimport calloc

from node cimport Node
from functions cimport fcn_sum, fcn_product, fcn_sin, fcn_sum_double_int
from types cimport DataType


cdef class Input(Node):
    cdef void* data
    cdef int data_size

    def __init__(
        self,
        data,
        data_type=DataType.TYPE_DOUBLE
    ):
        cdef int i

        super().__init__(data_type)

        self.data_size = len(data)
        self._convert_array_to_cython(data, data_type)

    cdef void _setup_cnode(self):
        self.cnode.inputs = cython.NULL
        self.cnode.input_sizes = cython.NULL
        self.cnode.input_types = cython.NULL
        self.cnode.input_count = 0
        self.cnode.data = self.data
        self.cnode.data_size = self.data_size
    
    def _convert_array_to_cython(self, data, data_type):
        cdef int i

        self.data = Node._allocate_data_array(self.data_type, len(data))

        if data_type == DataType.TYPE_INT:
            for i in range(self.data_size):
                (<int*>self.data)[i] = int(data[i])
        elif data_type == DataType.TYPE_FLOAT:
            for i in range(self.data_size):
                (<float*>self.data)[i] = float(data[i])
        else:
            for i in range(self.data_size):
                (<double*>self.data)[i] = float(data[i])


cdef class Sum(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fcn = fcn_sum


cdef class Product(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fcn = fcn_product


cdef class SumDoubleInt(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fcn = fcn_sum_double_int


cdef class Sin(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fcn = fcn_sin