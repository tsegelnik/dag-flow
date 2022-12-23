#!/usr/bin/env python

import numpy as np

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib.ViewConcat import ViewConcat
from dagflow.lib.Array import Array

debug = False

def test_ViewConcat_00():
    """Create four nodes: sum up three of them, multiply the result by the fourth
    Use graph context to create the graph.
    Use one-line code for connecting the nodes
    """
    array1 = np.arange(5.0)
    array2 = np.ones(shape=10, dtype='d')
    array3 = np.zeros(shape=12, dtype='d')-1
    array = np.concatenate((array1, array2, array3))
    with Graph(debug=debug, close=True) as graph:
        inputs = [Array('array', array) for array in (array1, array2, array3)]
        concat = ViewConcat("concat")

        inputs >> concat

    graph.print()

    assert all(initial.tainted==True for initial in inputs)
    assert concat.tainted==True

    result = concat.outputs['concat'].data
    # assert (result==array).all()
    # assert view.tainted==False
    # assert initial.tainted==False
    #
    # d1=initial.outputs[0]._data
    # d2=view.outputs[0]._data
    # assert (d1==d2).all()
    # d1[:]=-1
    # assert (d2==-1).all()
    #
    # initial.taint()
    # assert initial.tainted==True
    # assert view.tainted==True

    savegraph(graph, "output/test_ViewConcat_00.pdf")
