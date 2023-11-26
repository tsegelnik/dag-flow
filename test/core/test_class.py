#!/usr/bin/env python
from numpy import arange

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib import Array
from dagflow.lib import Product, Sum
from dagflow.printl import current_level, printl, set_prefix_function

set_prefix_function(lambda: "{:<2d} ".format(current_level()))


def test_00(testname, debug_graph):
    """Create four nodes: sum up three of them, multiply the result by the fourth
    Use graph context to create the graph.
    Use one-line code for connecting the nodes
    """
    array = arange(5)
    names = "n1", "n2", "n3", "n4"
    with Graph(debug=debug_graph) as graph:
        initials = [Array(name, array) for name in names]
        s = Sum("add")
        m = Product("mul")

    (initials[3], (initials[:3] >> s)) >> m

    graph.close()

    s.print()
    m.print()

    result = m.outputs["result"].data
    printl(result)

    savegraph(graph, "output/{testname}.pdf")
