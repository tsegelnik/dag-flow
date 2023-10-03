# to see output of this file you need use -s flag:
#       pytest -s ./test/test_profiling.py
import numpy as np

from dagflow.tools.profiling import IndividualProfiling

from dagflow.graph import Graph
from dagflow.lib.Array import Array
from dagflow.lib.MatrixProductDVDt import MatrixProductDVDt

left = np.array([[1, 2, 3], [3, 4, 5], [3, 4, 5]], dtype='d')
left2 = np.array([[1, 2, 3], [3, 4, 5], [3, 4, 5]], dtype='d')
square = np.diag(np.array([9, 4, 5], dtype='d'))
square2 = np.diag(np.array([9, 4, 5], dtype='d'))


with Graph(close=True) as graph:
    l_array = Array("Left_1", left)
    s_array = Array("Square_1", square)
    l_array_2 = Array("Left_2", left2)
    s_array_2 = Array("Square_2", square2)

    prod_1 = MatrixProductDVDt("MPDVDt_1")
    prod_2 = MatrixProductDVDt("MPDVDt_2")
    
    l_array >> prod_1.inputs["left"]
    s_array >> prod_1.inputs["square"]

    l_array_2 >> prod_2.inputs["left"]
    s_array_2 >> prod_2.inputs["square"]

nodes = graph._nodes

node_estimation = IndividualProfiling(nodes, n_runs=5000)
node_estimation.estimate_target_nodes().make_report()
