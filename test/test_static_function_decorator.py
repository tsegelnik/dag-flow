#!/usr/bin/env python


from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.input_extra import MissingInputAddOne
from dagflow.node_deco import NodeInstanceStatic
from dagflow.printl import current_level, printl, set_prefix_function
from dagflow.wrappers import *

set_prefix_function(
    lambda: "{:<2d} ".format(current_level()),
)
nodeargs = dict(typefunc=lambda: True, allocatable=False)
debug = False

call_counter = 0

with Graph(debug=debug) as graph:

    @NodeInstanceStatic(**nodeargs)
    def array():
        global call_counter
        call_counter += 1
        printl(f"Call array ({call_counter})")

    @NodeInstanceStatic(**nodeargs)
    def adder():
        global call_counter
        call_counter += 1
        printl(f"Call Adder ({call_counter})")

    @NodeInstanceStatic(**nodeargs)
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
