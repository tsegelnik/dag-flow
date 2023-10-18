# to see output of this file you need use -s flag:
#       pytest -s ./test/test_profiling.py
from collections import Counter

import numpy as np
import pytest

from dagflow.tools.profiling import IndividualProfiling, GroupProfiling

from dagflow.nodes import FunctionNode
from dagflow.graph import Graph
from dagflow.lib.Array import Array
from dagflow.lib.MatrixProductDVDt import MatrixProductDVDt
from dagflow.lib import Sum, Product
from dagflow.graphviz import GraphDot


class TestIndividual:
    n_runs = 1000
    def obtain_graph(self) -> Graph:
        left = np.array([[1, 2, 3], [3, 4, 5], [3, 4, 5]])
        square = np.diag(np.array([9, 4, 5]))
        square2 = np.diag(np.array([9, 4, 5]))

        with Graph(close=True) as graph:
            l_array = Array("Left_1", left)
            s_array = Array("Square_1", square)
            s_array_2 = Array("Square_2", square2)

            prod_1 = MatrixProductDVDt("MPDVDt_1")
            prod_2 = MatrixProductDVDt("MPDVDt_2")
            
            l_array >> prod_1.inputs["left"]
            s_array >> prod_1.inputs["square"]

            l_array >> prod_2.inputs["left"]
            s_array_2 >> prod_2.inputs["square"]

        return graph
    
    def check_inputs_taint(self, node: FunctionNode):
        return any(inp.tainted for inp in node.inputs)
    
    def test_estimate_node(self):
        g = self.obtain_graph()
        print(f"test IndividualProfiling.estimate_node (n_runs={self.n_runs}):")
        for node in g._nodes:
            print(f"{node.name} estimated with:",
                  IndividualProfiling.estimate_node(node, self.n_runs))
            assert self.check_inputs_taint(node) == False

    def test_estimate_target_nodes(self):
        g = self.obtain_graph()
        target_nodes = g._nodes
        profiling = IndividualProfiling(target_nodes, self.n_runs)
        profiling.estimate_target_nodes()
        assert hasattr(profiling, "_estimations_table")

    def test_make_report(self):
        g = self.obtain_graph()
        target_nodes = g._nodes
        profiling = IndividualProfiling(target_nodes, self.n_runs)
        profiling.estimate_target_nodes().make_report()
        profiling.make_report(group_by=None)
        profiling.make_report(group_by="name")
        with pytest.raises(ValueError):
            profiling.make_report(group_by="something wrong")
        with pytest.raises(ValueError):
            IndividualProfiling(target_nodes).make_report()


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

    source = array_nodes[0:2]
    sink = [prod1]
    profiling = GroupProfiling(source, sink)
    connected_nodes = [array_nodes[0], array_nodes[1], sum1, prod1]
    assert Counter(profiling._target_nodes) == Counter(connected_nodes)

    source = [sum1]
    sink = [prod2]
    profiling = GroupProfiling(source, sink)
    connected_nodes = [sum1, prod1, prod2]
    assert Counter(profiling._target_nodes) == Counter(connected_nodes)

    source = [array_nodes[2]]
    sink = [prod2]
    profiling = GroupProfiling(source, sink)
    result = profiling._target_nodes
    expected = [array_nodes[2], sum1, sum2, prod1, prod2]
    assert Counter(result) == Counter(expected)

    source = [array_nodes[4], array_nodes[0], array_nodes[1]]
    sink = [prod2]
    profiling = GroupProfiling(source, sink)
    result = profiling._target_nodes
    expected = source + [sum1, prod1, prod2]
    assert Counter(result) == Counter(expected)

    source = [array_nodes[4], array_nodes[0], array_nodes[1]]
    sink = [prod1]
    profiling = GroupProfiling(source, sink)
    result = profiling._target_nodes
    expected = source + [sum1, prod1]
    assert Counter(result) == Counter(expected)

    source = [array_nodes[0], array_nodes[1]]
    sink = [prod1, array_nodes[4]]
    with pytest.raises(ValueError):
        GroupProfiling(source, sink)

    source = [array_nodes[0], array_nodes[1]]
    sink = [prod2]
    profiling = GroupProfiling(source, sink, 500)
    result = profiling._target_nodes
    expected = source + [sum1, prod1] + sink
    assert Counter(result) == Counter(expected)
    
    profiling.estimate_framework_time()
    profiling.estimate_framework_time(True)
    profiling.make_report()
    profiling.make_report(top_n=5, group_by=None)

    graph_dot = GraphDot(graph)
    graph_dot.savegraph("output/test_profiling_graph_big.png")
    #TODO: split this test into small ones



     

