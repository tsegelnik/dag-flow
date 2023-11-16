# to see output of this file you need use -s flag:
#       pytest -s ./test/test_profiling.py
from collections import Counter

import numpy as np
import pytest

from dagflow.tools.profiling import Profiling, IndividualProfiling, FrameworkProfiling

from dagflow.nodes import FunctionNode
from dagflow.graph import Graph
from dagflow.lib.Array import Array
from dagflow.lib.MatrixProductDVDt import MatrixProductDVDt
from dagflow.lib import Sum, Product
from dagflow.graphviz import GraphDot


def graph_0() -> tuple[Graph, list[FunctionNode]]:
    with Graph(close=True) as graph:
        a0 = Array("A0", [8, 7, 13])
        a1 = Array("A1", [1, 2, 4])
        a2 = Array("A2", [12, 22, 121])
        a3 = Array("A3", [4, 3, 3])

        p0 = Product("P0")
        p1 = Product("P1")
        s0 = Sum("S0")

        (a0, a1) >> p0
        (a1, a2) >> p1
        (a2, a3) >> s0

        l_matrix = Array("L-Matrix", [[5, 2, 1], [1, 4, 12], [31, 7, 2]])
        s1 = Sum("S1 (diagonal of S-Matrix )")
        s2 = Sum("S2")

        (p0, p1, s0) >> s1
        (p1) >> s2

        mdvdt = MatrixProductDVDt("Matrix Product DVDt")
        s3 = Sum("S3")

        l_matrix >> mdvdt.inputs["left"]
        s1 >> mdvdt.inputs["square"]
        (s1, s2) >> s3
    s3['result'].data
    mdvdt['result'].data

    nodes = [a0, a1, a2, a3, p0, p1, s0, s1, s2, s3, l_matrix]
    return graph, nodes

def graph_1() -> tuple[Graph, list[FunctionNode]]:
    with Graph(close=True) as graph:
        array_nodes = [Array(f"A{i}", np.arange(i, i+3, dtype='f'))
                       for i in range(5)]
        s1 = Sum("S1")
        array_nodes[:3] >> s1
        
        s2 = Sum("S2")
        (array_nodes[2: 4]) >> s2 # array_3, array_4 >> sum_2
        
        p1 = Product("p1")
        (array_nodes[4], s1) >> p1

        p2 = Product("p2")
        (s2, p1) >> p2
    p2["result"].data

    nodes = [*array_nodes, s1, s2, p1, p2]
    return graph, nodes


class TestGraphsForTest:
    def test_invoke_and_save(self):
        graphs = [graph_0, graph_1]
        for i, g in enumerate(graphs):
            graph_dot = GraphDot(g()[0])
            graph_dot.savegraph(f"output/test_profiling_graph_{i}.png")


class TestBaseProfiling:
    def test_init_g0(self, monkeypatch):
        monkeypatch.setattr(Profiling, "__abstractmethods__", set())
        graph, nodes = graph_0()
        a0, a1, a2, a3, p0, p1, s0, s1, s2, s3, l_matrix = nodes
        
        target_nodes = [p1, s1, s2]
        profiling = Profiling(target_nodes, n_runs=10000)
        assert profiling._target_nodes == target_nodes 
        assert profiling._n_runs == 10000
        assert profiling._source == profiling._sink == []

        source, sink = [a2, a3], [s3]
        target_nodes = [a2, a3, s0, p1, s1, s2, s3]
        profiling = Profiling(source=source, sink=sink)
        
        assert Counter(profiling._target_nodes) == Counter(target_nodes)
        assert profiling._source == source
        assert profiling._sink == sink

        source, sink = [a0, a1, a2, a3, l_matrix], [s3]
        target_nodes = nodes
        profiling = Profiling(source=source, sink=sink)
        assert Counter(profiling._target_nodes) == Counter(target_nodes)

        source, sink = [a2, a3], [l_matrix]
        with pytest.raises(ValueError) as excinfo:
            Profiling(source=source, sink=sink)
        assert "nodes are unreachable" in str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            Profiling()
        assert "You shoud provide profiler with `target_nodes`" in str(excinfo.value)

    def test_init_g1(self, monkeypatch):
        monkeypatch.setattr(Profiling, "__abstractmethods__", set())
        graph, nodes = graph_1()
        a0, a1, a2, a3, a4, s1, s2, p1, p2 = nodes

        source, sink = [a4, s1], [p2]
        target_nodes = [a4, s1, p1, p2]
        profiling = Profiling(source=source, sink=sink)
        
        assert Counter(profiling._target_nodes) == Counter(target_nodes)
        assert profiling._source == source
        assert profiling._sink == sink
        
        source, sink = [a0, a1, a2, a3, a4], [p2]
        target_nodes = nodes
        profiling = Profiling(source=source, sink=sink)
        assert Counter(profiling._target_nodes) == Counter(target_nodes)

        source, sink = [a0, a2], [p1]
        target_nodes = [a0, a2, s1, p1]
        profiling = Profiling(source=source, sink=sink)
        assert Counter(profiling._target_nodes) == Counter(target_nodes)

        source, sink = [a0, a1], [s2]
        with pytest.raises(ValueError) as excinfo:
            Profiling(source=source, sink=sink)
        assert "nodes are unreachable" in str(excinfo.value)


class TestIndividual:
    n_runs = 1000

    def test_init(self):
        g, nodes = graph_0()
        a0, a1, a2, a3, p0, p1, s0, s1, s2, s3, l_matrix = nodes
        target_nodes=[a0, a1, s3, s2]
        profiling = IndividualProfiling(target_nodes)
        assert profiling._target_nodes == target_nodes
        
        source, sink = [a1, a2], [s2]
        target_nodes = [a1, a2, p1, s2]
        profiling = IndividualProfiling(target_nodes, source=source, sink=sink)

    def check_inputs_taint(self, node: FunctionNode):
        return any(inp.tainted for inp in node.inputs)
    
    def test_estimate_node_00(self):
        _, nodes = graph_0()
        print(f"(graph 0) IndividualProfiling.estimate_node (n_runs={self.n_runs}):")
        for node in nodes:
            print(f"\t{node.name} estimated with:",
                  IndividualProfiling.estimate_node(node, self.n_runs))
            assert self.check_inputs_taint(node) == False

    def test_estimate_node_01(self):
        _, nodes = graph_1()
        print(f"(graph 1) IndividualProfiling.estimate_node (n_runs={self.n_runs}):")
        for node in nodes:
            print(f"\t{node.name} estimated with:",
                  IndividualProfiling.estimate_node(node, self.n_runs))
            assert self.check_inputs_taint(node) == False

    def test_estimate_target_nodes_00(self):
        g, _ = graph_0()
        target_nodes = g._nodes
        profiling = IndividualProfiling(target_nodes, n_runs=self.n_runs)
        profiling.estimate_target_nodes()
        assert hasattr(profiling, "_estimations_table")

    def test_make_report_00(self):
        g, _ = graph_0()
        target_nodes = g._nodes
        profiling = IndividualProfiling(target_nodes, n_runs=self.n_runs)

        profiling.estimate_target_nodes().make_report()
        
        profiling.make_report(group_by=None)
        
        profiling.make_report(group_by="name")
        
        with pytest.raises(ValueError) as excinfo:
            profiling.make_report(group_by="something wrong")
        assert 'Invalid `group_by` name' in str(excinfo.value) 
        
        with pytest.raises(AttributeError) as excinfo:
            IndividualProfiling(target_nodes).make_report()
        assert 'No estimations found' in str(excinfo.value)

    def test_make_report_01(self):
        g, _ = graph_1()
        target_nodes = g._nodes
        profiling = IndividualProfiling(target_nodes, n_runs=self.n_runs)
        profiling.estimate_target_nodes()
        
        profiling.make_report(agg_funcs=['min', 'std', 'count'])

        # TODO: check for count
        # profiling.make_report(agg_funcs=['count', 'min', 'std'])

        with pytest.raises(ValueError) as excinfo:
            profiling.make_report(agg_funcs=['bad_function'])
        assert 'Invalid aggregate function' in str(excinfo.value)

        profiling.make_report(agg_funcs=['count', 'mean', 'min'], sort_by='min')
        profiling.make_report(agg_funcs=['count', 'mean', 'min'], sort_by='t_min')


# def test_group_pfoginling_01():
#     with Graph(close=True) as graph:
#         array_nodes = [Array(f"array_{i+1}", np.arange(i, i+3))
#                        for i in range(5)]
#         s1 = Sum("sum_1")
#         array_nodes[:3] >> s1
        
#         s2 = Sum("sum_2")
#         (array_nodes[2: 4]) >> s2 # array_3, array_4 >> sum_2
        
#         p1 = Product("prod_1")
#         (array_nodes[4], s1) >> p1

#         p2 = Product("prod_2")
#         (s2, p1) >> p2

#     p2["result"].data

#     source = array_nodes[0:2]
#     sink = [p1]
#     profiling = FrameworkProfiling(source=source, sink=sink)
#     connected_nodes = [array_nodes[0], array_nodes[1], s1, p1]
#     assert Counter(profiling._target_nodes) == Counter(connected_nodes)

#     source = [s1]
#     sink = [p2]
#     profiling = FrameworkProfiling(source=source, sink=sink)
#     connected_nodes = [s1, p1, p2]
#     assert Counter(profiling._target_nodes) == Counter(connected_nodes)

#     source = [array_nodes[2]]
#     sink = [p2]
#     profiling = FrameworkProfiling(source=source, sink=sink)
#     result = profiling._target_nodes
#     expected = [array_nodes[2], s1, s2, p1, p2]
#     assert Counter(result) == Counter(expected)

#     source = [array_nodes[4], array_nodes[0], array_nodes[1]]
#     sink = [p2]
#     profiling = FrameworkProfiling(source=source, sink=sink)
#     result = profiling._target_nodes
#     expected = source + [s1, p1, p2]
#     assert Counter(result) == Counter(expected)

#     source = [array_nodes[4], array_nodes[0], array_nodes[1]]
#     sink = [p1]
#     profiling = FrameworkProfiling(source=source, sink=sink)
#     result = profiling._target_nodes
#     expected = source + [s1, p1]
#     assert Counter(result) == Counter(expected)

#     source = [array_nodes[0], array_nodes[1]]
#     sink = [p1, array_nodes[4]]
#     with pytest.raises(ValueError):
#         FrameworkProfiling(source=source, sink=sink)

#     source = [array_nodes[0], array_nodes[1]]
#     sink = [p2]
#     profiling = FrameworkProfiling(source=source, sink=sink, n_runs=500)
#     result = profiling._target_nodes
#     expected = source + [s1, p1] + sink
#     assert Counter(result) == Counter(expected)
    
#     profiling.estimate_framework_time()
#     profiling.estimate_framework_time(True)
#     profiling.make_report()
#     profiling.make_report(top_n=5, group_by=None)

#     graph_dot = GraphDot(graph)
#     graph_dot.savegraph("output/test_profiling_graph_big.png")
#     #TODO: split this test into small ones

#     with pytest.raises(ValueError):
#         FrameworkProfiling([])



     

