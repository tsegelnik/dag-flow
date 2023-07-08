# to see output of this file you need use -s flag:
#       pytest -s ./test/test_profiling.py
import numpy as np

from dagflow.tools.profiling import Profiling

from dagflow.graph import Graph
from dagflow.lib.Array import Array
from dagflow.lib.MatrixProductDVDt import MatrixProductDVDt

left = np.array([[1, 2, 3], [3, 4, 5], [3, 4, 5]], dtype='d')
square = np.diag(np.array([9, 4, 5], dtype='d'))
square2 = np.diag(np.array([9, 4, 5], dtype='d'))

with Graph(close=True) as graph:
    l_array = Array("Left", left)
    s_array = Array("Square", square)
    s2_array = Array("Square2", square2)

    prod = MatrixProductDVDt("MatrixProductDVDt2d")
    l_array >> prod.inputs["left"]
    s_array >> prod.inputs["square"]

    prod2 = MatrixProductDVDt("MatrixProductDVDt2d2")
    l_array >> prod2.inputs["left"]
    s2_array >> prod2.inputs["square"]


profiling = Profiling()

for node in graph._nodes:
    profiling.estimate_node(node, times=1000000)

profiling.make_report()
