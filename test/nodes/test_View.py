#!/usr/bin/env python


from numpy import arange

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib.View import View
from dagflow.lib.Array import Array
from dagflow.wrappers import *

debug = False

def test_View_00():
    """Create four nodes: sum up three of them, multiply the result by the fourth
    Use graph context to create the graph.
    Use one-line code for connecting the nodes
    """
    array = arange(5.0)
    with Graph(debug=debug) as graph:
        initial = Array('array', array)
        view = View("view")

    initial >> view
    graph.close()

    assert view.tainted==True
    assert initial.tainted==True

    result = view.outputs.view.data
    assert (result==array).all()
    assert view.tainted==False
    assert initial.tainted==False

    assert initial.outputs[0]._data is view.outputs[0]._data
    initial.taint()
    assert view.tainted==True

    savegraph(graph, "output/class_00.pdf")
