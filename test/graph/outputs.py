from dagflow.lib import Array, Sum
from dagflow.graph import Graph
from dagflow.output import SettableOutput

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

