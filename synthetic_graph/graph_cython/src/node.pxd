from functions cimport CFcnType, CFcnDSType


ctypedef struct CNode:
    CNode** inputs
    int* input_sizes
    int* input_types
    int input_count

    void* data
    int data_size
    int data_type

    CFcnType* fcn
    int tainted


cdef class Node:
    cdef public list inputs
    cdef CNode* cnode
    cdef CFcnType* fcn
    cdef CFcnDSType* fcn_ds
    cdef int initialized
    cdef int data_type

    cdef CNode* _to_c_struct(self)
    cdef void _setup_cnode(self)
    @staticmethod
    cdef void* _run(CNode* node)
    @staticmethod
    cdef void* _allocate_data_array(int data_type, int data_size)
    @staticmethod
    cdef list _convert_array_to_python(void* result, int data_type, int data_size)

