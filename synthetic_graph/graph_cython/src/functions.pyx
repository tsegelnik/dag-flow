from libc.math cimport sin


cdef int fcn_ds_default(int* input_sizes):
    return input_sizes[0]


cdef void* fcn_default(void** inputs, int* input_sizes, int* input_types, int input_count, void* data, int data_type):
    return data


cdef void* fcn_sum(void** inputs, int* input_sizes, int* input_types, int input_count, void* data, int output_type):
    cdef int size = input_sizes[0]
    cdef int i, j
    
    for j in range(size):
        for i in range(input_count):
            (<double*>data)[j] += (<double**>inputs)[i][j]
    
    return data


cdef void* fcn_sum_double_int(void** inputs, int* input_sizes, int* input_types, int input_count, void* data, int output_type):
    cdef int size = input_sizes[0]
    cdef int i
    
    for i in range(size):
        (<double*>data)[i] += (<double**>inputs)[0][i]
        (<double*>data)[i] += (<int**>inputs)[1][i]
    
    return data


cdef void* fcn_product(void** inputs, int* input_sizes, int* input_types, int input_count, void* data, int output_type):
    cdef int size = input_sizes[0]
    cdef int i, j
    
    for j in range(size):
        for i in range(input_count):
            (<double*>data)[j] *= (<double**>inputs)[i][j]
    
    return data


cdef void* fcn_sin(void** inputs, int* input_sizes, int* input_types, int input_count, void* data, int output_type):
    cdef int size = input_sizes[0]
    cdef int i
    
    for i in range(size):
        (<double*>data)[i] = sin((<double**>inputs)[0][i])
    
    return data
