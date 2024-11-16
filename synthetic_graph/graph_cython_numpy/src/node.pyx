import cython
import functools
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as cnp

from functions cimport fcn_default

cnp.import_array()


cdef class Node:
    def __init__(self, inputs=None):
        self.inputs = inputs if inputs else []
        self.cnode = <CNode*>malloc(sizeof(CNode))
        self.fcn = fcn_default

    def __rshift__(self, other):
        other.inputs.append(self)

    @staticmethod
    cdef cnp.ndarray[double, ndim=1, mode="c"] _run(CNode* node):   
        cdef cnp.ndarray[double, ndim=2, mode="c"] input_results = np.zeros([
            node.input_count,
            max([s for s in node.input_sizes[:node.input_count]])
        ], dtype=np.double)
        cdef int i

        if node.data:
            input_results[0] = np.asarray([j for j in node.data[:node.data_size]], dtype=np.double) # TODO: check it
        else:
            for i in range(node.input_count):
                input_results[i] = Node._run(node.inputs[i])

        cdef cnp.ndarray[double, ndim=1, mode="c"] result = node.fcn(input_results)

        return result

    cdef CNode* _compile(self):
        cdef int i
        cdef int input_count = len(self.inputs)
        cdef CNode** input_nodes = <CNode**>malloc(sizeof(CNode*) * input_count)
        cdef int* input_sizes = <int*>malloc(sizeof(int) * input_count)

        cdef Node input_node
        for i in range(input_count):
            input_node = self.inputs[i]
            input_nodes[i] = <CNode*>input_node._compile()
            input_sizes[i] = input_node.get_size()

        self.cnode.inputs = input_nodes
        self.cnode.input_sizes = input_sizes
        self.cnode.input_count = input_count
        self.cnode.fcn = self.fcn
        self.cnode.data = cython.NULL
        self.cnode.data_size = 0

        return self.cnode

    def compile(self):
        self._compile()

    def run(self):
        return [i for i in Node._run(self.cnode)[:self.get_size()]]

    @functools.cache
    def get_size(self):
        return self.inputs[0].get_size()

    def __dealloc__(self):
        if self.cnode:
            free(self.cnode)



