from collections import Counter

import pytest

from dagflow.tools.profiling import NodeProfiler
from dagflow.node import Node

from test_helpers import graph_0, graph_1


n_runs = 1000

def test_init():
    g, nodes = graph_0()
    a0, a1, a2, a3, p0, p1, s0, s1, s2, s3, l_matrix, mdvdt = nodes
    target_nodes=[a0, a1, s3, s2]
    profiling = NodeProfiler(target_nodes)
    assert Counter(profiling._target_nodes) == Counter(target_nodes)

    sources, sinks = [a1, a2], [s2]
    target_nodes = [a1, a2, p1, s2]
    profiling = NodeProfiler(target_nodes, sources=sources, sinks=sinks)

def check_inputs_taint(node: Node):
    return any(inp.tainted for inp in node.inputs)

def test_estimate_node_g0():
    _, nodes = graph_0()
    print(f"(graph 0) NodeProfiler.estimate_node (n_runs={n_runs}):")
    for node in nodes:
        print(f"\t{node.name} estimated with:",
                NodeProfiler.estimate_node(node, n_runs))
        assert check_inputs_taint(node) == False

def test_estimate_node_g1():
    _, nodes = graph_1()
    print(f"(graph 1) NodeProfiler.estimate_node (n_runs={n_runs}):")
    for node in nodes:
        print(f"\t{node.name} estimated with:",
                NodeProfiler.estimate_node(node, n_runs))
        assert check_inputs_taint(node) == False

def test_estimate_target_nodes_g0():
    g, _ = graph_0()
    target_nodes = g._nodes
    profiling = NodeProfiler(target_nodes, n_runs=n_runs)
    profiling.estimate_target_nodes()
    assert hasattr(profiling, "_estimations_table")

def test_make_report_g0():
    g, _ = graph_0()
    target_nodes = g._nodes
    profiling = NodeProfiler(target_nodes, n_runs=n_runs)

    profiling.estimate_target_nodes().make_report()

    profiling.make_report(group_by=None)
    profiling.make_report(group_by="node")
    profiling.make_report(group_by="name")
    profiling.make_report(group_by="type")

    with pytest.raises(ValueError) as excinfo:
        profiling.make_report(group_by="something wrong")
    assert 'Invalid `group_by` name' in str(excinfo.value)

    with pytest.raises(AttributeError) as excinfo:
        NodeProfiler(target_nodes).make_report()
    assert 'No estimations found' in str(excinfo.value)

def test_make_report_g1():
    g, _ = graph_1()
    target_nodes = g._nodes
    profiling = NodeProfiler(target_nodes, n_runs=n_runs)
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

    profiling.make_report(agg_funcs=['count', 'single', 'min'],
                          sort_by='min')
    profiling.make_report(agg_funcs=['single', 'count', 'min'],
                          sort_by='t_single')
    profiling.make_report(agg_funcs=['single', 'count', 'min'],
                          sort_by='count')
    profiling.make_report(agg_funcs=['min', 'count', 'single'],
                          sort_by=None)

def test_print_report_g1_1():
    g, _ = graph_1()
    target_nodes = g._nodes
    profiling = NodeProfiler(target_nodes, n_runs=n_runs)
    profiling.estimate_target_nodes()

    profiling.print_report(agg_funcs=['min', 'single', 'count'], rows=500)
    profiling.print_report(agg_funcs=['min'], rows=1)
    profiling.print_report(group_by=None, rows=2)
    profiling.print_report(group_by=None, rows=20)
    profiling.print_report(agg_funcs=['single', 'count',
                                      'sum', 'percentage'],
                           sort_by='single')

def test_print_report_g1_2():
    g, _ = graph_1()
    target_nodes = g._nodes

    for i in range(2, 5):
        n_runs = 10 ** i
        profiling = NodeProfiler(target_nodes, n_runs=n_runs)
        profiling.estimate_target_nodes()
        profiling.print_report(agg_funcs=['single', 'count',
                                          'sum', 'percentage'])

def test_print_report_g1_3():
    g, _ = graph_1()
    target_nodes = g._nodes
    profiling = NodeProfiler(target_nodes, n_runs=n_runs)
    profiling.estimate_target_nodes()

    profiling.print_report(agg_funcs=['single', 'percentage', 'count'])
    profiling.print_report(agg_funcs=['count', 't_percentage'])
    profiling.make_report(agg_funcs=['t_percentage', 'count'])


