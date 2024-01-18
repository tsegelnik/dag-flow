# to see output of this file you need use -s flag:
#       pytest -s ./test/test_profiling.py
from collections import Counter

import numpy as np
import pytest
import types

from dagflow.tools.profiling.profiling import Profiling
from dagflow.tools.profiling import IndividualProfiling, FrameworkProfiling
from dagflow.tools.profiling import SleepyNode

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

    nodes = [a0, a1, a2, a3, p0, p1, s0, s1, s2, s3, l_matrix, mdvdt]
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
        a0, a1, a2, a3, p0, p1, s0, s1, s2, s3, l_matrix, mdvdt = nodes

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

        source, sink = [a0, a1, a2, a3, l_matrix], [s3, mdvdt]
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
        a0, a1, a2, a3, p0, p1, s0, s1, s2, s3, l_matrix, mdvdt = nodes
        target_nodes=[a0, a1, s3, s2]
        profiling = IndividualProfiling(target_nodes)
        assert Counter(profiling._target_nodes) == Counter(target_nodes)

        source, sink = [a1, a2], [s2]
        target_nodes = [a1, a2, p1, s2]
        profiling = IndividualProfiling(target_nodes, source=source, sink=sink)

    def check_inputs_taint(self, node: FunctionNode):
        return any(inp.tainted for inp in node.inputs)

    def test_estimate_node_g0(self):
        _, nodes = graph_0()
        print(f"(graph 0) IndividualProfiling.estimate_node (n_runs={self.n_runs}):")
        for node in nodes:
            print(f"\t{node.name} estimated with:",
                  IndividualProfiling.estimate_node(node, self.n_runs))
            assert self.check_inputs_taint(node) == False

    def test_estimate_node_g1(self):
        _, nodes = graph_1()
        print(f"(graph 1) IndividualProfiling.estimate_node (n_runs={self.n_runs}):")
        for node in nodes:
            print(f"\t{node.name} estimated with:",
                  IndividualProfiling.estimate_node(node, self.n_runs))
            assert self.check_inputs_taint(node) == False

    def test_estimate_target_nodes_g0(self):
        g, _ = graph_0()
        target_nodes = g._nodes
        profiling = IndividualProfiling(target_nodes, n_runs=self.n_runs)
        profiling.estimate_target_nodes()
        assert hasattr(profiling, "_estimations_table")

    def test_make_report_g0(self):
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

    def test_make_report_g1(self):
        g, _ = graph_1()
        target_nodes = g._nodes
        profiling = IndividualProfiling(target_nodes, n_runs=self.n_runs)
        profiling.estimate_target_nodes()

        profiling.make_report(agg_funcs=['min', 'std', 'count'])
        profiling.make_report(agg_funcs=['t_min', 't_std', 't_count'])
        profiling.make_report(agg_funcs=['t_mean', 't_percentage', 't_count'])
        profiling.make_report(agg_funcs=['median', '%_of_total'])
        profiling.make_report(agg_funcs=['count', 'min', 'std'])

        report = profiling.make_report(agg_funcs=['count', 'min', 'percentage'])
        assert 't_sum' not in report.columns
        report = profiling.make_report(agg_funcs=['sum', 'percentage'])
        assert 't_sum' in report.columns

        with pytest.raises(ValueError) as excinfo:
            profiling.make_report(agg_funcs=['bad_function'])
        assert 'Invalid aggregate function' in str(excinfo.value)

        profiling.make_report(agg_funcs=['count', 'single', 'min'], sort_by='min')
        profiling.make_report(agg_funcs=['single', 'count', 'min'], sort_by='t_single')
        profiling.make_report(agg_funcs=['single', 'count', 'min'], sort_by='count')
        profiling.make_report(agg_funcs=['min', 'count', 'single'], sort_by=None)

    def test_print_report_g1_1(self):
        g, _ = graph_1()
        target_nodes = g._nodes
        profiling = IndividualProfiling(target_nodes, n_runs=self.n_runs)
        profiling.estimate_target_nodes()

        profiling.print_report(agg_funcs=['min', 'single', 'count'], rows=500)
        profiling.print_report(agg_funcs=['min'], rows=1)
        profiling.print_report(group_by=None, rows=2)
        profiling.print_report(group_by=None, rows=20)
        profiling.print_report(agg_funcs=['single', 'count', 'sum', 'percentage'],
                               sort_by='single')

    def test_print_report_g1_1(self):
        g, _ = graph_1()
        target_nodes = g._nodes

        for i in range(2, 5):
            n_runs = 10 ** i
            profiling = IndividualProfiling(target_nodes, n_runs=n_runs)
            profiling.estimate_target_nodes()
            profiling.print_report(agg_funcs=['single', 'count',
                                                       'sum', 'percentage'])


class TestFrameworkProfiling:
    def test_init_g0(self):
        g, nodes = graph_0()
        a0, a1, a2, a3, p0, p1, s0, s1, s2, s3, l_matrix, mdvdt = nodes

        target_nodes = [a1, a2, s0, p1, p0, s1, s2, s3]
        profiling = FrameworkProfiling(target_nodes, n_runs=123)
        assert Counter(profiling._target_nodes) == Counter(target_nodes)
        assert profiling._n_runs == 123

        source, sink = [a1, a2], [s3]
        profiling = FrameworkProfiling(source=source, sink=sink)
        profiling.estimate_framework_time(append_results=True)
        profiling.print_report()
        assert Counter(profiling._target_nodes) == Counter(target_nodes)

    def test_reveal_source_sink_g0(self):
        _, nodes = graph_0()
        a0, a1, a2, a3, p0, p1, s0, s1, s2, s3, l_matrix, mdvdt = nodes

        source, sink = [a0, a1, a2, a3, l_matrix], [mdvdt, s3]
        profiling = FrameworkProfiling(nodes)
        assert Counter(profiling._source) == Counter(source)
        assert Counter(profiling._sink) == Counter(sink)

        target_nodes = [s1, s2, mdvdt, s3]
        source, sink = [s1, s2], [mdvdt, s3]
        profiling = FrameworkProfiling(target_nodes)
        assert Counter(profiling._source) == Counter(source)
        assert Counter(profiling._sink) == Counter(sink)
        # NOTE: this exception may be useful in future, but not now
        # target_nodes = [a1, a2, mdvdt]
        # with pytest.raises(ValueError) as excinfo:
        #     FrameworkProfiling(target_nodes)
        # assert 'no connections to other given nodes' in str(excinfo.value)

    def test__taint_nodes_g0(self):
        _, nodes = graph_0()

        profiling = FrameworkProfiling(nodes)
        profiling._taint_nodes()

        assert all(n.tainted for n in nodes)

    def test_make_fcns_empty_g0(self):
        _, nodes = graph_0()
        a0, a1, a2, a3, p0, p1, s0, s1, s2, s3, l_matrix, mdvdt = nodes

        profiling = FrameworkProfiling(nodes)
        profiling._make_fcns_empty()
        assert all(n.fcn == types.MethodType(FrameworkProfiling.fcn_no_computation, n) for n in nodes)

        profiling._taint_nodes()
        assert(a2.tainted == a1.tainted == p1.tainted == s2.tainted == True)
        s2.touch()
        assert(a2.tainted == a1.tainted == False)
        assert(p1.tainted == False)
        assert(s2.tainted == False)

        assert(s3.tainted == True)

    def test_underscore_estimate_framework_time_g0(self):
        _, nodes = graph_0()

        original_fcns = [n.fcn for n in nodes]
        profiling = FrameworkProfiling(nodes, n_runs=1000)

        results = profiling._estimate_framework_time()
        assert len(results) == profiling._n_runs

        final_fcns = [n.fcn for n in profiling._target_nodes]
        assert final_fcns == original_fcns

    def test_estimate_framework_time_g0(self):
        _, nodes = graph_0()

        FrameworkProfiling(nodes).estimate_framework_time()

        profiling = FrameworkProfiling(nodes)
        profiling.estimate_framework_time(append_results=True)
        profiling.estimate_framework_time(append_results=True)
        profiling.estimate_framework_time()

    def test_print_report_g0(self):
        _, nodes = graph_0()

        profiling = FrameworkProfiling(nodes, n_runs=1000)
        profiling.estimate_framework_time().print_report()
        profiling.estimate_framework_time(append_results=True)
        profiling.print_report()

        profiling.print_report(group_by=None)
        profiling.print_report(agg_funcs=['min', 'max', 'count'])

    def test_print_report_g1(self):
        _, nodes = graph_1()

        profiling = FrameworkProfiling(nodes, n_runs=1000)
        profiling.estimate_framework_time()
        profiling.print_report(agg_funcs=['single', 'sum', 'count'])


@pytest.mark.skip(reason="too slow to test every time")
class TestEstimationsTime:
    "Hint: use `pytest -s` to see estimations results"
    def test_one_sleepy_node(self):
        with Graph(close=True) as graph:
            sl = SleepyNode("SL0", sleep_time=0.25)
        sl['result'].data

        profiling = IndividualProfiling(graph._nodes, n_runs=4)
        res = profiling.estimate_node(sl, n_runs=4)
        print("SL0 (must be ≈ 1):", res)
        profiling.estimate_target_nodes()
        profiling.print_report()

    def _gen_graph(self, sleep_time: float):
        with Graph(close=True) as graph:
            sl0 = SleepyNode("SL0", sleep_time=sleep_time)
            sl1 = SleepyNode("SL1", sleep_time=sleep_time)
            sl2 = SleepyNode("SL2", sleep_time=sleep_time)
            (sl0, sl1) >> sl2
        sl2['result'].data
        return graph, [sl0, sl1, sl2]

    def test_three_sleepy_nodes(self):
        for x in (0.001, 0.1, 0.25, 0.5, 1):
            g, nodes = self._gen_graph(sleep_time=x)
            print("\nsleep_time =", x)
            profiling = IndividualProfiling(nodes, n_runs=5)
            profiling.estimate_target_nodes()
            profiling.print_report(group_by=None)

            fprofiling = FrameworkProfiling(nodes)
            fprofiling.estimate_framework_time().print_report()






