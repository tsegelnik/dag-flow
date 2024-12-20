import cython
from libc.stdlib cimport malloc, free, calloc

from functions cimport fcn_default, fcn_ds_default


cdef class Node:
    def __init__(self, inputs=None):
        self.inputs = inputs if inputs else []

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
        self.initialized = 1

        return self.cnode

    cdef void _setup_cnode(self):
        cdef int i
        cdef int input_count = len(self.inputs)
        cdef CNode** input_nodes = <CNode**> malloc(sizeof(CNode*) * input_count)
        cdef int* input_sizes = <int *> malloc(sizeof(int) * input_count)

        cdef Node input_node
        for i in range(input_count):
            input_node = self.inputs[i]
            input_nodes[i] = <CNode*>input_node._to_c_struct()
            input_sizes[i] = input_nodes[i].data_size

        cdef int data_size = self.fcn_ds(input_sizes)
        cdef double* data_array = <double*>calloc(sizeof(double), data_size)

        self.cnode.inputs = input_nodes
        self.cnode.input_sizes = input_sizes
        self.cnode.input_count = input_count
        self.cnode.data = data_array
        self.cnode.data_size = data_size

    def to_c_struct(self):
        self._to_c_struct()

    @staticmethod
    cdef double* _run(CNode* node):
        if not node.tainted:
            return node.data

        cdef double** input_results = <double**>malloc(node.input_count * sizeof(double*))
        cdef int i

        for i in range(node.input_count):
            input_results[i] = Node._run(node.inputs[i])

        cdef double* result = node.fcn(input_results, node.input_sizes, node.input_count, node.data)
        node.tainted = 0

        for i in range(node.input_count):
            if not node.inputs[i].data:
                free(input_results[i])

        free(input_results)
        return result

    def run(self):
        return [i for i in Node._run(self.cnode)[:self.cnode.data_size]]

    def __dealloc__(self):
        if self.cnode:
            free(self.cnode)



