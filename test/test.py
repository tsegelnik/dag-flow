#!/usr/bin/env python

from dagflow.input import Input
from dagflow.output import Output
from dagflow.node import FunctionNode
from dagflow.graph import Graph
from dagflow.wrappers import *

def test_01():
    i = Input('input', None)
    o = Output('output', None)

    o >> i

def test_02():
    n1 = FunctionNode('node1')
    n2 = FunctionNode('node2')

    n1._add_output('o1')
    n1._add_output('o2')

    n2._add_input('i1')
    n2._add_input('i2')
    n2._add_output('o1')

    n1 >> n2

def test_03():
    n1 = FunctionNode('node1')
    n2 = FunctionNode('node2')

    out = n1._add_output('o1')

    n2._add_input('i1')
    n2._add_output('o1')

    out >> n2

def test_04():
    n1 = FunctionNode('node1')
    n2 = FunctionNode('node2')

    out = n1._add_output('o1')

    n2._add_pair('i1', 'o1')

    final = out >> n2

def test_05():
    n1 = FunctionNode('node1')
    n2 = FunctionNode('node2')

    out1 = n1._add_output('o1')
    out2 = n1._add_output('o2')

    _, final = n2._add_pair('i1', 'o1')
    n2._add_input('i2')

    (out1, out2) >> n2

def test_06():
    n1 = FunctionNode('node1')
    n2 = FunctionNode('node2')

    out1 = n1._add_output('o1')
    out2 = n1._add_output('o2')

    _, final = n2._add_pair('i1', 'o1')
    n2._add_input('i2')

    (out1, out2) >> n2

def test_07():
    g = Graph()
    n1 = g.add_node('node1')
    n2 = g.add_node('node2')
    g._wrap_fcns(toucher, printer)

    out1 = n1._add_output('o1')
    out2 = n1._add_output('o2')

    _, final = n2._add_pair('i1', 'o1')
    n2._add_input('i2')

    (out1, out2) >> n2

    final.data

def test_08():
    g = Graph()
    n1 = g.add_node('node1')
    n2 = g.add_node('node2')
    n3 = g.add_node('node3')
    g._wrap_fcns(toucher, printer)

    out1 = n1._add_output('o1')
    out2 = n1._add_output('o2')

    _, out3 = n2._add_pair('i1', 'o1')
    n2._add_input('i2')

    _, final = n3._add_pair('i1', 'o1')

    (out1, out2) >> n2
    out3 >> n3

    print()
    final.data

    print('Taint n2')
    n2.taint()
    final.data

    print('Taint n3')
    n3.taint()
    final.data

if __name__ == "__main__":
    test_01()
    test_02()
    test_03()
    test_04()
    test_05()
    test_06()
    test_07()
    test_08()
