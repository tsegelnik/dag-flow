# to see output of this file you need use -s flag:
#       pytest -s ./test/test_profiling.py
import numpy as np

from dagflow.tools.profiling import Profiling, GroupProfiling

from dagflow.graph import Graph
from dagflow.lib.Array import Array
from dagflow.lib.ElSumSq import ElSumSq
# from dagflow.lib import ElSumSq
from dagflow.lib.MatrixProductDVDt import MatrixProductDVDt

left = np.array([[1, 2, 3], [3, 4, 5], [3, 4, 5]], dtype='d')
left2 = np.array([[1, 2, 3], [3, 4, 5], [3, 4, 5]], dtype='d')
square = np.diag(np.array([9, 4, 5], dtype='d'))
square2 = np.diag(np.array([9, 4, 5], dtype='d'))

rng = np.random.default_rng()

arrays_data = rng.random(size=(10, 20), dtype='float64')
print(arrays_data[0])


with Graph(close=True) as graph:
    array_nodes = [Array(f"A_{i}", arrays_data[i]) for i in range(10)]
    elsum = ElSumSq("ElSumSq")
    array_nodes >> elsum


print(list(elsum.outputs.iter_data()))
profiling = Profiling(n_runs=10000).estimate_graph(graph)
profiling.make_report()

profiling = GroupProfiling()
profiling.estimate_group_with_empty_fcn()

