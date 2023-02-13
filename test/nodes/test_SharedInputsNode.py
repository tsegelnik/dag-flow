#!/usr/bin/env python
from numpy import arange

from dagflow.graph import Graph
from dagflow.lib.SharedInputsNode import SharedInputsNode
from dagflow.lib.View import View
from dagflow.lib.Array import Array
from dagflow.exception import ConnectionError
from dagflow.graphviz import savegraph
from pytest import raises

def test_SharedInputsNode_00(debug_graph=False):
    array = arange(12.0).reshape(3, 4)
    with Graph(debug=debug_graph, close=True) as graph:
        initial = Array("array 1", array, mode='store_weak')
        initial2 = Array("array 2", array, mode='store_weak')
        initial3 = Array("array 3", array, mode='store_weak')
        sharedinput = SharedInputsNode("shared input")
        view = View("view")

        initial >> sharedinput
        initial2 >> sharedinput >> view
        initial3 >> sharedinput

    output_array = initial.outputs[0]
    output_array2 = initial2.outputs[0]
    output_array3 = initial3.outputs[0]
    input_sharedinput = sharedinput.inputs[0]
    output_sharedinput = sharedinput.outputs[0]
    output_view = view.outputs[0]

    assert output_array.allocatable == True
    assert output_array2.allocatable == True
    assert output_array3.allocatable == True
    assert output_sharedinput.allocatable == False
    assert input_sharedinput.allocatable == True
    assert output_view.allocatable == False

    assert input_sharedinput._own_data is sharedinput._data
    assert output_sharedinput._data is sharedinput._data
    assert output_view._data is sharedinput._data

    assert initial.tainted == True
    assert initial2.tainted == True
    assert initial3.tainted == True
    assert sharedinput.tainted == True
    assert view.tainted == True

    assert (output_sharedinput.data == array).all()
    assert (output_view.data == array).all()

    assert initial.tainted == False
    assert initial2.tainted == False
    assert initial3.tainted == False
    assert sharedinput.tainted == False
    assert view.tainted == False

    initial._data[:] = 1
    initial.taint()
    assert initial.tainted == True
    assert initial2.tainted == True
    assert initial3.tainted == True
    assert sharedinput.tainted == True
    assert view.tainted == True
    assert (output_sharedinput.data == 1).all()
    assert (output_view.data == 1).all()
    assert (output_array.data == 1).all()
    assert (output_array2.data == 1).all()
    assert (output_array3.data == 1).all()
    assert initial.tainted == False
    assert initial2.tainted == False
    assert initial3.tainted == False

    initial2._data[:] = 2
    initial2.taint()
    assert initial.tainted == True
    assert initial2.tainted == True
    assert initial3.tainted == True
    assert sharedinput.tainted == True
    assert view.tainted == True
    assert (output_sharedinput.data == 2).all()
    assert (output_view.data == 2).all()
    assert (output_array.data == 2).all()
    assert (output_array2.data == 2).all()
    assert (output_array3.data == 2).all()
    assert initial.tainted == False
    assert initial2.tainted == False
    assert initial3.tainted == False

    view.touch()
    savegraph(graph, "output/test_SharedInputsNode_00.png")

def test_SharedInputsNode_01():
    array = arange(12.0).reshape(3, 4)
    with Graph() as graph:
        initial = Array("array 1", array, mode='store_weak')
        initial2 = Array("array 2", array, mode='store_weak')
        sharedinput = SharedInputsNode("shared input")
        sharedinput2 = SharedInputsNode("shared input 2")

        initial >> sharedinput
        initial2 >> sharedinput

        with raises(ConnectionError):
            sharedinput >> sharedinput2

    savegraph(graph, "output/test_SharedInputsNode_01.png")

def test_SharedInputsNode_02():
    array = arange(12.0).reshape(3, 4)
    with Graph(close=False) as graph:
        initial = Array("array 1", array, mode='store')
        sharedinput = SharedInputsNode("shared input")
        sharedinput2 = SharedInputsNode("shared input 2")
        view = View("view")

        with raises(ConnectionError):
            initial >> sharedinput

        with raises(ConnectionError):
            view >> sharedinput

        with raises(ConnectionError):
            sharedinput >> sharedinput2

    savegraph(graph, "output/test_SharedInputsNode_02.png")
