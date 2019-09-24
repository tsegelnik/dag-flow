#!/usr/bin/env python

from __future__ import print_function
from dagflow.node_deco import NodeClass
from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.input_extra import MissingInputAddOne
import numpy as N

# Node functions
@NodeClass(output='array')
def Array(self, inputs, outputs, node):
    """Creates a note with single data output with predefined array"""
    outputs[0].data = N.arange(5, dtype='d')

@NodeClass(missing_input_handler=MissingInputAddOne(output_fmt='result'))
def Adder(self, inputs, outputs, node):
    """Adds all the inputs together"""
    out = None
    for input in inputs:
        if out is None:
            out=outputs[0].data = input.data
        else:
            out+=input.data

@NodeClass(missing_input_handler=MissingInputAddOne(output_fmt='result'))
def Multiplier(self, inputs, outputs, node):
    """Multiplies all the inputs together"""
    out = None
    for input in inputs:
        if out is None:
            out = outputs[0].data = input.data
        else:
            out*=input.data

# The actual code
with Graph() as graph:
    (in1, in2, in3, in4) = [Array(name) for name in ['n1', 'n2', 'n3', 'n4']]
    s = Adder('add')
    m = Multiplier('mul')

(in1, in2, in3) >> s
(in4, s) >> m

print('Result is:', m.outputs.result.data)
savegraph(graph, 'output/dagflow_example.png')

