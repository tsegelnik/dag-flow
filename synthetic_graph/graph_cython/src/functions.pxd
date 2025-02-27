ctypedef void* CFcnType(void** inputs, int* input_sizes, int* input_types, int input_count, void* data, int data_type)

cdef CFcnType fcn_default
cdef CFcnType fcn_sum
cdef CFcnType fcn_sum_double_int
cdef CFcnType fcn_product
cdef CFcnType fcn_sin

ctypedef int CFcnDSType(int* input_sizes)
cdef CFcnDSType fcn_ds_default
