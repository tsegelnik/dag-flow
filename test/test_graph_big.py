#!/usr/bin/env python
from dagflow.graph import Graph
from dagflow.graphviz import GraphDot
from dagflow.printl import current_level, set_prefix_function
from dagflow.lib.Dummy import Dummy
from dagflow.wrappers import *

set_prefix_function(lambda: "{:<2d} ".format(current_level()))

counter = 0


def test_graph_big_01():
    """Create a graph of nodes and test evaluation features"""
    g = Graph()
    label = None

    def plot(suffix=""):
        global counter
        d = GraphDot(g)
        newlabel = label and label + suffix or suffix
        if newlabel is not None:
            d.set_label(newlabel)
        d.savegraph("output/test_graph_big_{:03d}.png".format(counter))
        counter += 1

    def plotter(fcn, node, inputs, outputs):
        plot(f"[start evaluating {node.name}]")
        fcn(node, inputs, outputs)
        plot(f"[done evaluating {node.name}]")

    with g:
        A1 = Dummy("A1")
        A2 = Dummy("A2", auto_freeze=True, label="{name}|frozen")
        A3 = Dummy("A3", immediate=True, label="{name}|immediate")
        B = Dummy("B")
        C1 = Dummy("C1")
        C2 = Dummy("C2")
        D = Dummy("D")
        E = Dummy("E")
        F = Dummy("F")
        H = Dummy("H")
        P = Dummy("P", immediate=True, label="{name}|immediate")

    g._wrap_fcns(toucher, printer, plotter)

    A1._add_output("o1", allocatable=False)
    A2._add_output("o1", allocatable=False)
    P._add_output("o1", allocatable=False)
    A3._add_pair("i1", "o1", output_kws={"allocatable": False})
    B._add_pair(
        ("i1", "i2", "i3", "i4"),
        ("o1", "o2"),
        output_kws={"allocatable": False},
    )
    C1._add_output("o1", allocatable=False)
    C2._add_output("o1", allocatable=False)
    D._add_pair("i1", "o1", output_kws={"allocatable": False})
    D._add_pair("i2", "o2", output_kws={"allocatable": False})
    H._add_pair("i1", "o1", output_kws={"allocatable": False})
    _, other = F._add_pair("i1", "o1", output_kws={"allocatable": False})
    _, final = E._add_pair("i1", "o1", output_kws={"allocatable": False})

    (P >> A3)
    (A1, A2, A3, D[:1]) >> B
    B >> (E, H)
    (C1, C2) >> D
    D[:, 1] >> F

    g.print()
    g.close()

    label = "Initial graph state."
    plot()

    label = "Read E..."
    plot()
    plot()
    plot()
    final.data
    label = "Done reading E."
    plot()

    label = "Taint D."
    plot()
    plot()
    plot()
    D.taint()
    plot()
    label = "Read F..."
    other.data
    label = "Done reading F."
    plot()

    label = "Read E..."
    plot()
    plot()
    plot()
    final.data
    label = "Done reading E."
    plot()

    label = "Taint A2."
    plot()
    plot()
    plot()
    A2.taint()
    plot()
    label = "Read E..."
    plot()
    final.data
    label = "Done reading E."
    plot()

    label = "Unfreeze A2 (tainted)."
    plot()
    plot()
    plot()
    A2.unfreeze()
    plot()
    label = "Read E..."
    plot()
    final.data
    label = "Done reading E."
    plot()

    label = "Unfreeze A2 (not tainted)."
    plot()
    plot()
    plot()
    A2.unfreeze()
    plot()
    label = "Read E..."
    plot()
    final.data
    label = "Done reading E."
    plot()

    label = "Taint P"
    plot()
    plot()
    plot()
    P.taint()
    plot()
    label = "Read E..."
    plot()
    final.data
    label = "Done reading E."
    plot()

    label = "Invalidate P"
    plot()
    plot()
    plot()
    P.invalid = True
    plot()

    label = "Validate P"
    plot()
    plot()
    plot()
    P.invalid = False
    plot()
    label = "Read E..."
    plot()
    final.data
    label = "Done reading E."
    plot()
