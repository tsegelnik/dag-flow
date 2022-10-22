#!/usr/bin/env python

from __future__ import print_function

import numpy as N

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.input_extra import MissingInputAddOne
from dagflow.node_deco import NodeInstanceStatic
from dagflow.printl import current_level, printl, set_prefix_function
from dagflow.wrappers import *

set_prefix_function(
    lambda: "{:<2d} ".format(current_level()),
)

call_counter = 0

with Graph() as graph:

    @NodeInstanceStatic()
    def array():
        global call_counter
        call_counter += 1
        printl(f"Call array ({call_counter})")

    @NodeInstanceStatic()
    def adder():
        global call_counter
        call_counter += 1
        printl(f"Call Adder ({call_counter})")

    @NodeInstanceStatic()
    def multiplier():
        global call_counter
        call_counter += 1
        printl(f"Call Multiplier ({call_counter})")


def test_00():
    array >> adder
    (array, adder) >> multiplier

    graph._wrap_fcns(dataprinter, printer)
    graph.close()

    result = multiplier.outputs[0].data
    printl(result)

    savegraph(graph, "output/decorators_static_graph_00.pdf")


if __name__ == "__main__":
    test_00()
