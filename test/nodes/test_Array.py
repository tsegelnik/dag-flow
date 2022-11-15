#!/usr/bin/env python

from numpy import arange

from dagflow.graph import Graph
from dagflow.lib.Array import Array
from dagflow.wrappers import *

debug = False

def test_Array_00():
    array = arange(12.0).reshape(3,4)
    with Graph(debug=debug) as graph:
        arr = Array('array', array)
    graph.close()

    assert arr.tainted==True

    result = arr.outputs[0].data
    assert (result==array).all()
    assert arr.tainted==False
