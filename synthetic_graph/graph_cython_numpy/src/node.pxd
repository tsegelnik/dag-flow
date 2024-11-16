cimport numpy as cnp

from functions cimport CFcnType


ctypedef struct CNode:
    CNode** inputs
    int* input_sizes
    int input_count
    CFcnType* fcn
    double* data
    int data_size


cdef class Node:
    cdef public list inputs
    cdef CNode* cnode
    cdef CFcnType* fcn

    cdef CNode* _compile(self)

    @staticmethod
    cdef cnp.ndarray[double, ndim=1, mode="c"] _run(CNode* node)
