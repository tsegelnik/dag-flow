ctypedef double* CFcnType(double** inputs, int* input_sizes, int input_count, double* data)

cdef CFcnType fcn_default
cdef CFcnType fcn_sum
cdef CFcnType fcn_product
cdef CFcnType fcn_sin
