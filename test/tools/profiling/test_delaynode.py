# to see estimations results need use -s flag:
#       pytest -s ./test/tools/test_delaynode.py

import pytest

from dagflow.graph import Graph

from dagflow.tools.profiling import NodeProfiler, FrameworkProfiler
from dagflow.tools.profiling import DelayNode

@pytest.mark.skip(reason="too slow to test every time")
def test_one_delay_node():
    with Graph(close=True) as graph:
        sl = DelayNode("SL0", sleep_time=0.25)
    sl['result'].data

    profiling = NodeProfiler(graph._nodes, n_runs=4)
    res = profiling.estimate_node(sl, n_runs=4)
    print("SL0 (must be ≈ 1):", res)
    profiling.estimate_target_nodes()
    profiling.print_report()

def _gen_graph(sleep_time: float):
    with Graph(close=True) as graph:
        sl0 = DelayNode("SL0", sleep_time=sleep_time)
        sl1 = DelayNode("SL1", sleep_time=sleep_time)
        sl2 = DelayNode("SL2", sleep_time=sleep_time)
        (sl0, sl1) >> sl2
    sl2['result'].data
    return graph, [sl0, sl1, sl2]

@pytest.mark.skip(reason="too slow to test every time")
def test_three_delay_nodes():
    for x in (0.001, 0.1, 0.25, 0.5, 1):
        g, nodes = _gen_graph(sleep_time=x)
        print("\nsleep_time =", x)
        profiling = NodeProfiler(nodes, n_runs=5)
        profiling.estimate_target_nodes()
        profiling.print_report(group_by=None)

        fprofiling = FrameworkProfiler(nodes)
        fprofiling.estimate_framework_time().print_report()
