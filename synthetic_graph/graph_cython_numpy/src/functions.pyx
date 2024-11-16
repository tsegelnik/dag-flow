from libc.math cimport sin, cosh
import numpy as np
cimport numpy as cnp


cdef cnp.ndarray[double, ndim=1, mode="c"] fcn_integration(cnp.ndarray[double, ndim=2, mode="c"] np_inputs):
    return np.array([np.trapz(np_inputs[0], np_inputs[1])], dtype=np.double)

cdef cnp.ndarray[double, ndim=1, mode="c"] fcn_sum(cnp.ndarray[double, ndim=2, mode="c"] np_inputs):
    return np.sum(np_inputs, axis=0)

cdef cnp.ndarray[double, ndim=1, mode="c"] fcn_product(cnp.ndarray[double, ndim=2, mode="c"] np_inputs):
    return np.prod(np_inputs, axis=0)

cdef cnp.ndarray[double, ndim=1, mode="c"] fcn_sin(cnp.ndarray[double, ndim=2, mode="c"] np_inputs):
    return np.sin(np.clip(np_inputs[0], -10, 10))

cdef cnp.ndarray[double, ndim=1, mode="c"] fcn_cosh(cnp.ndarray[double, ndim=2, mode="c"] np_inputs):
    return np.cosh(np.clip(np_inputs[0], -10, 10))

cdef cnp.ndarray[double, ndim=1, mode="c"] fcn_default(cnp.ndarray[double, ndim=2, mode="c"] np_inputs):
    return np_inputs[0]