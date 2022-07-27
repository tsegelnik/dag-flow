#!/usr/bin/env python

from __future__ import print_function

from dagflow.graph import Graph
from dagflow.graphviz import GraphDot
from dagflow.printl import current_level, set_prefix_function
from dagflow.wrappers import *

set_prefix_function(
    lambda: "{:<2d} ".format(current_level()),
)


def test_01():
    """Simple test of the graph plotter"""
    g = Graph()
    n1 = g.add_node("node1")
    n2 = g.add_node("node2")
    n3 = g.add_node("node3")
    g._wrap_fcns(toucher, printer)

    in1, out1 = n1._add_pair("i1", "o1")
    out2 = n1._add_output("o2")

    _, out3 = n2._add_pair("i1", "o1")
    n2._add_input("i2")

    _, final = n3._add_pair("i1", "o1")

    print(f"{out1=}, {out2=}")
    (out1, out2) >> n2
    out3 >> n3
    g.close()

    d = GraphDot(g)
    d.savegraph("output/test1_00.png")


def test_02():
    """Simple test of the graph plotter"""
    g = Graph()
    n1 = g.add_node("node1")
    n2 = g.add_node("node2")
    n3 = g.add_node("node3")
    g._wrap_fcns(toucher, printer)

    out1 = n1._add_output("o1")
    out2 = n1._add_output("o2")

    _, out3 = n2._add_pair("i1", "o1")
    n2._add_input("i2")

    _, final = n3._add_pair("i1", "o1")

    (out1, out2) >> n2
    out3 >> n3
    g.close()

    d = GraphDot(g)
    d.savegraph("output/test2_00.png")

    final.data
    d = GraphDot(g)
    d.savegraph("output/test2_01.png")


def test_02a():
    """Simple test of the graph plotter"""
    g = Graph()
    n1 = g.add_node("node1")
    n2 = g.add_node("node2")
    n3 = g.add_node("node3")
    n4 = g.add_node("node4")
    g._wrap_fcns(toucher, printer)

    out1 = n1._add_output("o1")

    in2, out2 = n2._add_pair("i1", "o1")
    in3, out3 = n3._add_pair("i1", "o1")
    in4, out4 = n4._add_pair("i1", "o1")

    out1.repeat() >> (in2, in3, in4)
    g.close()

    d = GraphDot(g)
    d.savegraph("output/test2a_00.png")

    print(out4.data)
    d = GraphDot(g)
    d.savegraph("output/test2a_01.png")


if __name__ == "__main__":
    test_01()
    test_02()
    test_02a()
