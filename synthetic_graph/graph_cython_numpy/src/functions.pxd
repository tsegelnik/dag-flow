cimport numpy as cnp

ctypedef cnp.ndarray[double, ndim=1, mode="c"] CFcnType(cnp.ndarray[double, ndim=2, mode="c"] np_inputs)

cdef CFcnType fcn_default
cdef CFcnType fcn_sum
cdef CFcnType fcn_product
cdef CFcnType fcn_sin
cdef CFcnType fcn_cosh
cdef CFcnType fcn_integration
