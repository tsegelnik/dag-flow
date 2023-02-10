#!/usr/bin/env python

from numpy import arange

from dagflow.graph import Graph
from dagflow.lib.Array import Array
from dagflow.lib.Sum import Sum

debug = False

def test_Array_00():
    array = arange(12.0).reshape(3,4)
    with Graph(close=True):
        arr1 = Array('array', array, mode='store')
        arr2 = Array('array', array, mode='store_weak')
        arr3 = Array('array', array, mode='fill')

    assert arr1.tainted==True
    assert arr2.tainted==True
    assert arr3.tainted==True

    out1 = arr1.outputs['array']
    out2 = arr2.outputs['array']
    out3 = arr3.outputs['array']

    assert out1.owns_buffer == True
    assert out2.owns_buffer == False
    assert out3.owns_buffer == True

    assert out1.allocatable == False
    assert out2.allocatable == True
    assert out3.allocatable == True

    assert (out1._data==array).all()
    assert (out2._data==array).all()
    assert (out3._data==0.0).all()

    result1 = arr1.get_data(0)
    result2 = arr2.get_data(0)
    result3 = arr3.get_data(0)

    assert (result1==array).all()
    assert (result2==array).all()
    assert (result3==array).all()
    assert arr1.tainted==False
    assert arr2.tainted==False
    assert arr3.tainted==False

def test_Array_01_set():
	value = 123.
	array_in = (value, )
	array_alt = (value+1, )
	va = Array('test', array_in)
	sm = Sum('sum')
	va >> sm
	va.close()
	sm.close()

	output = va.outputs[0]
	output2 = sm.outputs[0]

	assert va.tainted==True
	assert sm.tainted==True
	assert output.data[0]==value
	assert output2.data[0]==value
	assert va.tainted==False
	assert sm.tainted==False

	assert va.set(array_in, check_taint=True)==False
	assert va.tainted==False
	assert sm.tainted==False
	assert (output.data==array_in).all()
	assert (output2.data==array_in).all()

	assert va.set(array_in)==True
	assert va.tainted==False
	assert sm.tainted==True
	assert (output.data==array_in).all()
	assert (output2.data==array_in).all()
	assert va.tainted==False
	assert sm.tainted==False

	assert va.set(array_alt, check_taint=True)==True
	assert va.tainted==False
	assert sm.tainted==True
	assert (output.data==array_alt).all()
	assert (output2.data==array_alt).all()
	assert va.tainted==False
	assert sm.tainted==False
