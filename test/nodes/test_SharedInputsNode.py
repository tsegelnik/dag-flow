#!/usr/bin/env python
from numpy import arange

from dagflow.graph import Graph
from dagflow.lib.SharedInputsNode import SharedInputsNode
from dagflow.lib.Array import Array
from dagflow.lib.Sum import Sum
from dagflow.graphviz import savegraph

def test_SharedInputsNode_00(debug_graph):
    array = arange(12.0).reshape(3, 4)
    with Graph(debug=debug_graph, close=True) as graph:
        initial = Array("array 1", array)
        initial2 = Array("array 2", array)
        sum = Sum("copy 1")
        sum2 = Sum("copy 2")
        view = SharedInputsNode("view")

        initial >> sum >> view
        initial2 >> sum2 >> view

    graph.print()

    output_array = initial.outputs[0]
    input_sum = sum.inputs[0]
    output_sum = sum.outputs[0]
    input_sum2 = sum2.inputs[0]
    output_sum2 = sum2.outputs[0]
    input_view = view.inputs[0]
    output_view = view.outputs[0]

    assert output_array.allocatable == False
    assert output_sum.allocatable == True
    assert output_sum2.allocatable == True
    assert output_view.allocatable == False
    assert input_sum.allocatable == False
    assert input_sum2.allocatable == False
    assert input_view.allocatable == True

    assert input_view._own_data is view._data
    assert output_view._data is view._data
    assert output_sum._data is view._data
    assert output_sum2._data is view._data

    assert initial.tainted == True
    assert initial2.tainted == True
    assert sum.tainted == True
    assert sum2.tainted == True
    assert view.tainted == True

    assert (output_view.data == array).all()

    assert initial.tainted == False
    assert initial2.tainted == False
    assert sum.tainted == False
    assert sum2.tainted == False
    assert view.tainted == False

    initial._data[:] = 1
    initial.taint()
    assert initial.tainted == True
    assert initial2.tainted == False
    assert sum.tainted == True
    assert sum2.tainted == False
    assert view.tainted == True
    assert (output_view.data == 1).all()
    assert (output_sum.data == 1).all()
    assert (output_sum2.data == 1).all()

    initial2._data[:] = 2
    initial2.taint()
    assert initial.tainted == False
    assert initial2.tainted == True
    assert sum.tainted == False
    assert sum2.tainted == True
    assert view.tainted == True
    assert (output_view.data == 2).all()
    assert (output_sum.data == 2).all()
    assert (output_sum2.data == 2).all()

    savegraph(graph, "output/test_SharedInputsNode_00.png")
