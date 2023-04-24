#!/usr/bin/env python

from pytest import raises
import numpy as np

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib.ViewConcat import ViewConcat
from dagflow.lib.View import View
from dagflow.lib.NormalizeCorrelatedVars2 import NormalizeCorrelatedVars2
from dagflow.lib.Array import Array
from dagflow.exception import ConnectionError

import pytest

debug = False

@pytest.mark.parametrize('closemode', ['graph', 'recursive'])
def test_ViewConcat_00(closemode):
    """Create four nodes: sum up three of them, multiply the result by the fourth
    Use graph context to create the graph.
    Use one-line code for connecting the nodes
    """
    closegraph = closemode=='graph'

    array1 = np.arange(5.0)
    array2 = np.ones(shape=10, dtype='d')
    array3 = np.zeros(shape=12, dtype='d')-1
    array = np.concatenate((array1, array2, array3))
    arrays = (array1, array2, array3)
    n1, n2, _ = (a.size for a in arrays)
    with Graph(debug=debug, close=closegraph) as graph:
        inputs = [Array('array', array, mode='fill') for array in arrays]
        concat = ViewConcat("concat")
        view = View("view")

        inputs >> concat >> view

    if not closegraph:
        view.close()

    graph.print()

    assert all(initial.tainted==True for initial in inputs)
    assert concat.tainted==True
    assert view.tainted==True

    result = concat.get_data()
    result_view = view.get_data()
    assert (result==array).all()
    assert (result_view==array).all()
    assert concat.tainted==False
    assert view.tainted==False
    assert all(i.tainted==False for i in inputs)

    data1, data2, data3 = (i.get_data(0) for i in inputs)
    datac = concat.get_data(0)
    datav = view.get_data(0)
    assert all(data1==datac[:data1.size])
    assert all(data2==datac[n1:n1+data2.size])
    assert all(data3==datac[n1+n2:n1+n2+data3.size])

    data1[2]=-1
    data2[:]=-1
    data3[::2]=-2
    assert all(data1==datac[:data1.size])
    assert all(data2==datac[n1:n1+data2.size])
    assert all(data3==datac[n1+n2:n1+n2+data3.size])
    assert all(data1==datav[:data1.size])
    assert all(data2==datav[n1:n1+data2.size])
    assert all(data3==datav[n1+n2:n1+n2+data3.size])

    inputs[1].taint()
    assert concat.tainted==True
    assert view.tainted==True

    view.touch()
    savegraph(graph, "output/test_ViewConcat_00.png")

def test_ViewConcat_01():
    with Graph() as graph:
        concat = ViewConcat("concat")
        concat2 = ViewConcat("concat 2")
        view = View('view')
        normnode = NormalizeCorrelatedVars2('normvars')

        with raises(ConnectionError):
            view >> concat

        with raises(ConnectionError):
            normnode.outputs[0] >> concat

        with raises(ConnectionError):
            concat >> normnode.inputs[0]

        with raises(ConnectionError):
            concat >> concat2

    savegraph(graph, "output/test_ViewConcat_01.png")
