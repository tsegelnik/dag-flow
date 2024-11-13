import cython
import functools
from libc.stdlib cimport malloc, free

from functions cimport fcn_default


cdef class Node:
    def __init__(self, inputs=None):
        self.inputs = inputs if inputs else []
        self.cnode = <CNode*>malloc(sizeof(CNode))
        self.fcn = fcn_default

    def __rshift__(self, other):
        other.inputs.append(self)

    cdef CNode* _compile(self):
        cdef int i
        cdef int input_count = len(self.inputs)
        cdef CNode** input_nodes = <CNode**> malloc(sizeof(CNode*) * input_count)
        cdef int * input_sizes = <int *> malloc(sizeof(int) * input_count)


        cdef Node input_node
        for i in range(input_count):
            input_node = self.inputs[i]
            input_nodes[i] = <CNode*>input_node._compile()
            input_sizes[i] = input_node.get_size()

        cdef int size = self.get_size()
        cdef double* data_array = <double*>malloc(sizeof(double) * size)

        self.cnode.inputs = input_nodes
        self.cnode.input_sizes = input_sizes
        self.cnode.input_count = input_count
        self.cnode.fcn = self.fcn
        self.cnode.data = data_array

        return self.cnode

    def compile(self):
        self._compile()

    @staticmethod
    cdef double* _run(CNode* node):
        cdef double** input_results = <double**>malloc(node.input_count * sizeof(double*))
        cdef int i

        for i in range(node.input_count):
            input_results[i] = Node._run(node.inputs[i])

        cdef double* result = node.fcn(input_results, node.input_sizes, node.input_count, node.data)

        for i in range(node.input_count):
            if not node.inputs[i].data:
                free(input_results[i])

        free(input_results)
        return result

    def run(self):
        return [i for i in Node._run(self.cnode)[:self.get_size()]]

    @functools.cache
    def get_size(self):
        return self.inputs[0].get_size()

    def __dealloc__(self):
        if self.cnode:
            free(self.cnode)



