#!/usr/bin/env python

from numpy import arange

from dagflow.graph import Graph
from dagflow.lib.Array import Array
from dagflow.wrappers import *

debug = False

def test_Array_00():
    array = arange(12.0).reshape(3,4)
    with Graph(close=True) as graph:
        arr1 = Array('array', array, mode='store')
        arr2 = Array('array', array, mode='fill')

    assert arr1.tainted==True
    assert arr2.tainted==True

    out1 = arr1.outputs['array']
    out2 = arr2.outputs['array']

    assert (out1._data==array).all()
    assert (out2._data==0.0).all()

    result1 = arr1.get_data(0)
    result2 = arr2.get_data(0)

    assert (result1==array).all()
    assert (result2==array).all()
    assert arr1.tainted==False
    assert arr2.tainted==False
