#!/usr/bin/env python

from __future__ import print_function

from numpy import arange

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.input_extra import MissingInputAddOne
from dagflow.lib import Array, Product, Sum
from dagflow.node import FunctionNode
from dagflow.printl import current_level, printl, set_prefix_function
from dagflow.wrappers import *

set_prefix_function(lambda: "{:<2d} ".format(current_level()))
debug = False


def test_00():
    """Create four nodes: sum up three of them, multiply the result by the fourth
    Use graph context to create the graph.
    Use one-line code for connecting the nodes
    """
    array = arange(5)
    with Graph(debug=debug) as graph:
        initials = [Array(name, array) for name in ("n1", "n2", "n3", "n4")]
        s = Sum("add")
        m = Product("mul")

    (initials[3], (initials[:3] >> s)) >> m

    graph._wrap_fcns(dataprinter, printer)
    graph.close()

    result = m.outputs.result.data
    printl(result)

    savegraph(graph, "output/class_00.pdf")


if __name__ == "__main__":
    test_00()
