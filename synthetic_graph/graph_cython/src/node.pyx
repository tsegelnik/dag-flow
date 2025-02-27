import cython
from libc.stdlib cimport malloc, free, calloc

from functions cimport fcn_default, fcn_ds_default
from types cimport DataType


cdef class Node:
    def __init__(self, data_type=DataType.TYPE_DOUBLE, inputs=None):
        self.inputs = inputs if inputs else []
        self.data_type = data_type

        self.cnode = <CNode*>malloc(sizeof(CNode))
        self.initialized = 0

        self.fcn = fcn_default
        self.fcn_ds = fcn_ds_default

    def __rshift__(self, other):
        other.inputs.append(self)

    cdef CNode* _to_c_struct(self):
        if self.initialized:
            return self.cnode

        self._setup_cnode()
        self.cnode.fcn = self.fcn
        self.cnode.tainted = 1
        self.cnode.data_type = self.data_type
        self.initialized = 1

        return self.cnode

    cdef void _setup_cnode(self):
        cdef int i
        cdef int input_count = len(self.inputs)
        cdef CNode** input_nodes = <CNode**>malloc(sizeof(CNode*) * input_count)
        cdef int* input_sizes = <int*>malloc(sizeof(int) * input_count)
        cdef int* input_types = <int*>malloc(sizeof(int) * input_count)

        cdef Node input_node
        for i in range(input_count):
            input_node = self.inputs[i]
            input_nodes[i] = <CNode*>input_node._to_c_struct()
            input_sizes[i] = input_nodes[i].data_size
            input_types[i] = input_nodes[i].data_type

        cdef int data_size = self.fcn_ds(input_sizes) 
        
        cdef void* data_array = Node._allocate_data_array(self.data_type, data_size)

        self.cnode.inputs = input_nodes
        self.cnode.input_sizes = input_sizes
        self.cnode.input_count = input_count
        self.cnode.input_types = input_types
        self.cnode.data = data_array
        self.cnode.data_size = data_size

    def to_c_struct(self):
        self._to_c_struct()

    @staticmethod
    cdef void* _run(CNode* node):
        if not node.tainted:
            return node.data

        cdef void** input_results = <void**>malloc(node.input_count * sizeof(void*))
        cdef int i

        for i in range(node.input_count):
            input_results[i] = Node._run(node.inputs[i])

        cdef void* result = node.fcn(input_results, node.input_sizes, node.input_types, node.input_count, node.data, node.data_type)
        node.tainted = 0

        for i in range(node.input_count):
            if not node.inputs[i].data:
                free(input_results[i])

        free(input_results)
        return result

    def run(self):
        cdef void* result = Node._run(self.cnode)
        return Node._convert_array_to_python(result, self.data_type, self.cnode.data_size)
    
    @staticmethod
    cdef list _convert_array_to_python(void* result, int data_type, int data_size):
        cdef int i
        output = []
        
        if data_type == DataType.TYPE_INT:
            for i in range(data_size):
                output.append((<int*>result)[i])
        elif data_type == DataType.TYPE_FLOAT:
            for i in range(data_size):
                output.append((<float*>result)[i])
        else:
            for i in range(data_size):
                output.append((<double*>result)[i])
        
        return output

    @staticmethod
    cdef void* _allocate_data_array(int data_type, int data_size):
        if data_type == DataType.TYPE_INT:
            return calloc(data_size, sizeof(int))
        elif data_type == DataType.TYPE_FLOAT:
            return calloc(data_size, sizeof(float))
        else:
            return calloc(data_size, sizeof(double))

    def __dealloc__(self):
        if self.cnode:
            if self.cnode.data:
                free(self.cnode.data)
            if self.cnode.inputs:
                free(self.cnode.inputs)
            if self.cnode.input_sizes:
                free(self.cnode.input_sizes)
            free(self.cnode)
