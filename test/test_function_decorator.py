#!/usr/bin/env python


from numpy import arange

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.input_extra import MissingInputAddOne
from dagflow.node_deco import NodeClass, NodeInstance
from dagflow.printl import current_level, printl, set_prefix_function
from dagflow.wrappers import *

set_prefix_function(lambda: "{:<2d} ".format(current_level()))
nodeargs = dict(typefunc=lambda: True, allocatable=False)
debug = False


@NodeClass(output="array", **nodeargs)
def Array(node, inputs, outputs):
    """Creates a note with single data output with predefined array"""
    outputs[0].data = arange(5, dtype="d")


@NodeClass(missing_input_handler=MissingInputAddOne(output_fmt="result"), **nodeargs)
def Adder(node, inputs, outputs):
    """Adds all the inputs together"""
    out = outputs[0].data = inputs[0].data.copy()
    for input in inputs[1:]:
        out += input.data


@NodeClass(missing_input_handler=MissingInputAddOne(output_fmt="result"), **nodeargs)
def Multiplier(node, inputs, outputs):
    """Multiplies all the inputs together"""
    out = outputs[0].data = inputs[0].data.copy()
    for input in inputs[1:]:
        out *= input.data


def test_00():
    """Create four nodes: sum up three of them, multiply the result by the fourth
    Use Graph methods to build the graph
    """
    graph = Graph(debug=debug)
    in1 = graph.add_node("n1", nodeclass=Array, **nodeargs)
    in2 = graph.add_node("n2", nodeclass=Array, **nodeargs)
    in3 = graph.add_node("n3", nodeclass=Array, **nodeargs)
    in4 = graph.add_node("n4", nodeclass=Array, **nodeargs)
    s = graph.add_node("add", nodeclass=Adder, **nodeargs)
    m = graph.add_node("mul", nodeclass=Multiplier, **nodeargs)

    (in1, in2, in3) >> s
    (in4, s) >> m

    graph._wrap_fcns(dataprinter, printer)
    graph.close()

    printl(m.outputs.result.data)
    savegraph(graph, "output/decorators_graph_00.pdf")


def test_01():
    """Create four nodes: sum up three of them, multiply the result by the fourth
    Use graph context to create the graph.
    Use one-line code for connecting the nodes
    """
    with Graph(debug=debug) as graph:
        initials = [
            Array(name, **nodeargs) for name in ("n1", "n2", "n3", "n4")
        ]
        s = Adder("add", **nodeargs)
        m = Multiplier("mul", **nodeargs)

    (initials[3], (initials[:3] >> s)) >> m

    graph._wrap_fcns(dataprinter, printer)
    graph.close()

    printl(m.outputs.result.data)
    savegraph(graph, "output/decorators_graph_01.pdf")


def test_02():
    """Create four nodes: sum up three of them, multiply the result by the fourth
    Use graph context to create the graph.
    Use one-line code for connecting the nodes.
    Use NodeInstance decorator to convert functions directly to node instances.
    """
    with Graph(debug=debug) as graph:
        initials = [
            Array(name, **nodeargs) for name in ("n1", "n2", "n3", "n4")
        ]

        @NodeInstance(
            name="add",
            class_kwargs=dict(
                missing_input_handler=MissingInputAddOne(output_fmt="result")
            ),
            **nodeargs,
        )
        def s(node, inputs, outputs):
            out = outputs[0].data = inputs[0].data
            for input in inputs[1:]:
                out += input.data

        @NodeInstance(
            name="mul",
            class_kwargs=dict(
                missing_input_handler=MissingInputAddOne(output_fmt="result")
            ),
            **nodeargs,
        )
        def m(node, inputs, outputs):
            out = outputs[0].data = inputs[0].data
            for input in inputs[1:]:
                out *= input.data

    (initials[3], (initials[:3] >> s)) >> m

    graph._wrap_fcns(dataprinter, printer)
    graph.close()

    printl(m.outputs.result.data)
    savegraph(graph, "output/decorators_graph_02.pdf")
