from functions cimport CFcnType


ctypedef struct CNode:
    CNode** inputs
    int* input_sizes
    int input_count
    CFcnType* fcn
    double* data


cdef class Node:
    cdef public list inputs
    cdef CNode* cnode
    cdef CFcnType* fcn

    cdef CNode* _compile(self)

    @staticmethod
    cdef double* _run(CNode* node)
