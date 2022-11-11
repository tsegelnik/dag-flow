from dagflow.lib import VariableArray

def test_VariableArray_01():
	value = 123.
	array_in = (value, )
	array_alt = (value+1, )
	va = VariableArray('test', array_in)
	va.close()

	output = va.outputs.array

	assert va.tainted==True
	assert output.data[0]==value
	assert va.tainted==False

	assert va.set(array_in, check_taint=True)==False
	assert va.tainted==False

	assert va.set(array_in)==True
	assert va.tainted==True
	output.data
	assert va.tainted==False

	assert va.set(array_alt, check_taint=True)==True
	assert va.tainted==True
	output.data
	assert va.tainted==False
