from dagflow.graph import Graph
from dagflow.lib import Array
from dagflow.lib import Sum


def test_Output_01():
    value = 123.0
    array_in = (value,)
    array_alt = (value + 1,)
    with Graph(close_on_exit=True):
        va = Array("test", array_in)
        s = Sum("add")
        va >> s
    out = s.outputs[0]
    assert out.tainted
    out.set(array_alt)
    assert not out.tainted
    assert (s.outputs[0].data == array_alt).all()


def test_Output_02():
    """Test Output, Node.invalidate_parents()"""
    value = 123.0
    array_in = (value,)
    array_alt = (value + 1,)
    with Graph(close_on_exit=True):
        va = Array("test", array_in)
        sm1 = Sum("add 1")
        sm2 = Sum("add 2")
        va >> sm1 >> sm2

    output1 = va.outputs[0]
    output2 = sm1.outputs[0]
    output3 = sm2.outputs[0]

    assert va.tainted
    assert sm1.tainted
    assert sm2.tainted
    assert not va.invalid
    assert not sm1.invalid
    assert not sm2.invalid
    assert output3.data == array_in
    assert not va.tainted
    assert not sm1.tainted
    assert not sm2.tainted

    output2.set(array_alt)
    assert va.tainted
    assert not sm1.tainted
    assert sm2.tainted
    assert va.invalid
    assert not sm1.invalid
    assert not sm2.invalid
    assert output2.data == array_alt
    assert output3.data == array_alt
    assert output1.data != array_alt

    va.invalid = False
    assert va.tainted
    assert sm1.tainted
    assert sm2.tainted
    assert not va.invalid
    assert not sm1.invalid
    assert not sm2.invalid
    assert output3.data == array_in
    assert output2.data == array_in
    assert output1.data == array_in
