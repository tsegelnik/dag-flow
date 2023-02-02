#!/usr/bin/env python

from numpy import arange

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib.View import View
from dagflow.lib.Array import Array
from dagflow.lib.NormalizeCorrelatedVars import NormalizeCorrelatedVars
from dagflow.wrappers import *

debug = False

def test_NormalizeCorrelatedVars_00():
    array = arange(5.0)
    with Graph(close=True) as graph:
        initial1 = Array('array1', array)
        initial2 = Array('array2', array)
        norm = NormalizeCorrelatedVars('norm')

        (initial1, initial2) >> norm

        # view = View("view")
        # view2 = View("view2")
        #
        # initial >> view >> view2

    # assert initial.tainted==True
    # assert view.tainted==True
    # assert view2.tainted==True
    #
    # result = view.get_data()
    # result2 = view2.get_data()
    # assert (result==array).all()
    # assert (result2==array).all()
    # assert view.tainted==False
    # assert view2.tainted==False
    # assert initial.tainted==False
    #
    # d1=initial.outputs[0]._data
    # d2=view.outputs[0]._data
    # d3=view2.outputs[0]._data
    # assert (d1==d2).all()
    # assert (d1==d3).all()
    # d1[:]=-1
    # assert (d2==-1).all()
    # assert (d3==-1).all()
    #
    # initial.taint()
    # assert initial.tainted==True
    # assert view.tainted==True
    # assert view2.tainted==True
    #
    # view2.touch()
    savegraph(graph, "output/test_View_00.png")
