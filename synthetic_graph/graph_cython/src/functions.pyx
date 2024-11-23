from libc.math cimport sin


cdef double* fcn_default(double** inputs, int* input_sizes, int input_count, double* data):
    return data


cdef double* fcn_sum(double** inputs, int* input_sizes, int input_count, double* data):
    cdef int size = input_sizes[0]
    cdef int i, j

    for i in range(input_count):
        for j in range(size):
            data[j] += inputs[i][j]

    return data


cdef double* fcn_product(double** inputs, int* input_sizes, int input_count, double* data):
    cdef int size = input_sizes[0]
    cdef int i, j

    for i in range(input_count):
        for j in range(size):
            data[j] *= inputs[i][j]

    return data


cdef double* fcn_sin(double** inputs, int* input_sizes, int input_count, double* data):
    cdef double* input = inputs[0]
    cdef int i

    for i in range(input_sizes[0]):
        data[i] = sin(input[i])

    return data
