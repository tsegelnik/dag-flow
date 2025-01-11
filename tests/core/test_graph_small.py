from numpy import arange

from dagflow.core.graph import Graph
from dagflow.plot.graphviz import savegraph
from dagflow.lib.common import Array
from dagflow.lib.arithmetic import Product, Sum


def test_graph_small(testname, debug_graph):
    """Create four arrays: sum up three of them, multiply the result by the fourth
    Use graph context to create the graph.
    """
    array = arange(4)
    names = "n1", "n2", "n3", "n4"
    with Graph(debug=debug_graph) as graph:
        initials = [Array(name, array) for name in names]
        s = Sum("add")
        m = Product("mul")

        initials[:-1] >> s
        initials[-1] >> m
        s >> m

    graph.close()

    s.print()
    m.print()

    result = m.outputs["result"].data
    print("Evaluation result:", result)

    savegraph(graph, f"output/{testname}.pdf")
