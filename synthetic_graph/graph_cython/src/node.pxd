from functions cimport CFcnType


ctypedef struct CNode:
    CNode** inputs
    int* input_sizes
    int input_count
    CFcnType* fcn
    double* data
    int tainted


cdef class Node:
    cdef public list inputs
    cdef CNode* cnode
    cdef CFcnType* fcn
    cdef int initialized

    cdef CNode* _to_c_struct(self)
    
    cdef void _setup_cnode(self)

    @staticmethod
    cdef double* _run(CNode* node)
