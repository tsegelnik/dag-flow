from dagflow.lib import VariableArray, Sum

def test_VariableArray_01():
	value = 123.
	array_in = (value, )
	array_alt = (value+1, )
	va = VariableArray('test', array_in)
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
