# to see output of this file you need use -s flag:
#       pytest -s ./test/test_profiling.py
import numpy as np

from dagflow.tools.profiling import IndividualProfiling, GroupProfiling

from dagflow.graph import Graph
from dagflow.lib.Array import Array
from dagflow.lib.MatrixProductDVDt import MatrixProductDVDt
from dagflow.lib.Dummy import Dummy
from dagflow.lib import Sum, Product
from dagflow.graphviz import GraphDot

# left = np.array([[1, 2, 3], [3, 4, 5], [3, 4, 5]], dtype='d')
# left2 = np.array([[1, 2, 3], [3, 4, 5], [3, 4, 5]], dtype='d')
# square = np.diag(np.array([9, 4, 5], dtype='d'))
# square2 = np.diag(np.array([9, 4, 5], dtype='d'))


# with Graph(close=True) as graph:
#     l_array = Array("Left_1", left)
#     s_array = Array("Square_1", square)
#     l_array_2 = Array("Left_2", left2)
#     s_array_2 = Array("Square_2", square2)

#     prod_1 = MatrixProductDVDt("MPDVDt_1")
#     prod_2 = MatrixProductDVDt("MPDVDt_2")
    
#     l_array >> prod_1.inputs["left"]
#     s_array >> prod_1.inputs["square"]

#     l_array >> prod_2.inputs["left"]
#     s_array_2 >> prod_2.inputs["square"]

# nodes = graph._nodes

# node_estimation = IndividualProfiling(nodes, n_runs=50000)
# node_estimation.estimate_target_nodes().make_report()

# node_estimation = IndividualProfiling(nodes, n_runs=50000)
# node_estimation.estimate_target_nodes().make_report(group_by=None)

# node_estimation = IndividualProfiling(nodes, n_runs=50000)
# node_estimation.estimate_target_nodes().make_report(group_by="name")

# nodes = [l_array, prod_1]

# node_estimation = GroupProfiling([l_array], [prod_1])



def test_group_pfoginling_01():
    with Graph(close=True) as graph:
        array_nodes = [Array(f"array_{i+1}", np.arange(i, i+3))
                       for i in range(5)]
        sum1 = Sum("sum_1")
        array_nodes[:3] >> sum1
        
        sum2 = Sum("sum_2")
        (array_nodes[2: 4]) >> sum2 # array_3, array_4 >> sum_2
        
        prod1 = Product("prod_1")
        (array_nodes[4], sum1) >> prod1

        prod2 = Product("prod_2")
        (sum2, prod1) >> prod2

    prod2["result"].data

    graph_dot = GraphDot(graph)
    graph_dot.savegraph("output/test_profiling_graph_big_1.png")

    source = array_nodes[0:2]
    sink = [prod1]
    profiling = GroupProfiling(source, sink)
    connected_nodes = [array_nodes[0], array_nodes[1], sum1, prod1]
    assert len(profiling._target_nodes) == len(connected_nodes)
    assert set(profiling._target_nodes) == set(connected_nodes)

    source = [sum1]
    sink = [prod2]
    profiling = GroupProfiling(source, sink)
    connected_nodes = [sum1, prod1, prod2]
    assert len(profiling._target_nodes) == len(connected_nodes)
    assert set(profiling._target_nodes) == set(connected_nodes)

    source = [array_nodes[2]]
    sink = [prod2]
    profiling = GroupProfiling(source, sink)
    result = profiling._target_nodes
    expected = [array_nodes[2], sum1, sum2, prod1, prod2]
    # print('\n', *result, sep='\n')
    # print('\n', *expected, sep='\n')
    assert len(result) == len(expected)
    assert set(result) == set(expected)

    source = [array_nodes[4], array_nodes[0], array_nodes[1]]
    sink = [prod2]
    profiling = GroupProfiling(source, sink)
    result = profiling._target_nodes
    expected = source + [sum1, prod1, prod2]
    # print('\n', *result, sep='\n')
    # print('\n', *expected, sep='\n')
    assert len(result) == len(expected)
    assert set(result) == set(expected)

    source = [array_nodes[4], array_nodes[0], array_nodes[1]]
    sink = [prod1]
    profiling = GroupProfiling(source, sink)
    result = profiling._target_nodes
    expected = source + [sum1, prod1]
    # print('\n', *result, sep='\n')
    # print('\n', *expected, sep='\n')
    assert len(result) == len(expected)
    assert set(result) == set(expected)

    graph_dot = GraphDot(graph)
    graph_dot.savegraph("output/test_profiling_graph_big_2.png")



     

