from dagflow.lib.Array import Array
from dagflow.lib.Sum import Sum
from dagflow.graph import Graph
from dagflow.output import SettableOutput
from dagflow.exception import CriticalError
import pytest

def test_SettableOutput_01():
	value = 123.
	array_in = (value, )
	array_alt = (value+1, )
	with Graph() as g:
		va = Array('test', array_in)
		s = Sum('add')
		va >> s
	g.close()

	va.taint()
	newout = SettableOutput.take_over(va.outputs.array)
	newout.set(array_alt)

	assert va.outputs.array is newout
	assert s.inputs[0].parent_output is newout
	assert s.outputs.result.data==array_alt

def test_SettableOutput_02():
	"""Test SettableOutput, Node.invalidate_parents()"""
	value = 123.
	array_in = (value, )
	array_alt = (value+1, )
	with Graph() as g:
		va = Array('test', array_in)
		sm1 = Sum('add 1')
		sm2 = Sum('add 2')
		va >> sm1 >> sm2
	g.close()

	output1 = va.outputs[0]
	output2 = sm1.outputs[0]
	output3 = sm2.outputs[0]

	assert va.tainted==True
	assert sm1.tainted==True
	assert sm2.tainted==True
	assert va.invalid==False
	assert sm1.invalid==False
	assert sm2.invalid==False
	assert output3.data==array_in
	assert va.tainted==False
	assert sm1.tainted==False
	assert sm2.tainted==False

	newout = SettableOutput.take_over(sm1.outputs[0])
	assert va.tainted==False
	assert sm1.tainted==False
	assert sm2.tainted==False
	assert va.invalid==False
	assert sm1.invalid==False
	assert sm2.invalid==False
	assert output3.data==array_in

	newout.set(array_alt)
	assert va.tainted==True
	assert sm1.tainted==False
	assert sm2.tainted==True
	assert va.invalid==True
	assert sm1.invalid==False
	assert sm2.invalid==False
	assert output2.data==array_alt
	assert output3.data==array_alt
	with pytest.raises(CriticalError):
		output1.data==array_alt

	va.invalid = False
	assert va.tainted==True
	assert sm1.tainted==True
	assert sm2.tainted==True
	assert va.invalid==False
	assert sm1.invalid==False
	assert sm2.invalid==False
	assert output3.data==array_in
	assert output2.data==array_in
	assert output1.data==array_in

