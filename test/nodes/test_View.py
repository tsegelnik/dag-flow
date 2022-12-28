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

    assert initial.tainted==True
    assert view.tainted==True

    result = view.outputs['view'].data
    assert (result==array).all()
    assert view.tainted==False
    assert initial.tainted==False

    d1=initial.outputs[0]._data
    d2=view.outputs[0]._data
    assert (d1==d2).all()
    d1[:]=-1
    assert (d2==-1).all()

    initial.taint()
    assert initial.tainted==True
    assert view.tainted==True

    savegraph(graph, "output/class_00.pdf")
