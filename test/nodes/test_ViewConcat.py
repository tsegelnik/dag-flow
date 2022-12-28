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
    arrays = (array1, array2, array3)
    n1, n2, n3 = (a.size for a in arrays)
    with Graph(debug=debug, close=True) as graph:
        inputs = [Array('array', array, mode='fill') for array in arrays]
        concat = ViewConcat("concat")

        inputs >> concat

    graph.print()

    assert all(initial.tainted==True for initial in inputs)
    assert concat.tainted==True

    result = concat.outputs['concat'].data
    assert (result==array).all()
    assert concat.tainted==False
    assert all(i.tainted==False for i in inputs)

    data1, data2, data3 = (i.get_data(0) for i in inputs)
    datac = concat.get_data(0)
    assert all(data1==datac[:data1.size])
    assert all(data2==datac[n1:n1+data2.size])
    assert all(data3==datac[n1+n2:n1+n2+data3.size])

    data1[2]=-1
    data2[:]=-1
    data3[::2]=-2
    assert all(data1==datac[:data1.size])
    assert all(data2==datac[n1:n1+data2.size])
    assert all(data3==datac[n1+n2:n1+n2+data3.size])

    inputs[1].taint()
    assert concat.tainted==True

    concat.touch()
    savegraph(graph, "output/test_ViewConcat_00.png")
