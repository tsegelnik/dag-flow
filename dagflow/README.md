# Summary

[![python](https://img.shields.io/badge/python-3.10-purple.svg)](https://www.python.org/)
[![pipeline](https://git.jinr.ru/dag-computing/dag-flow/badges/master/pipeline.svg)](https://git.jinr.ru/dag-computing/dag-flow/commits/master)
[![coverage report](https://git.jinr.ru/dag-computing/dag-flow/badges/master/coverage.svg)](https://git.jinr.ru/dag-computing/dag-flow/-/commits/master)
<!--- Uncomment here after adding docs!
[![pages](https://img.shields.io/badge/pages-link-white.svg)](http://dag-computing.pages.jinr.ru/dag-flow)
-->

DAGFlow is python implementation of dataflow programming with lazy graph evaluation.

Main goals:
*  Lazy evaluated directed acyclic graph
*  Concise connection syntax
*  Plotting with graphviz
*  Flexibility. The goal of DAGFlow is not to be efficient, but rather flexible.

Here is an animation, showing the process of the graph evaluation

![Image](example/graph_evaluation.gif)

# Minimal example
An example of small, graph calculating the formula (n1 + n2 + n3) * n4 may be 
found in the [example](example/example.py):
```python
#!/usr/bin/env python

from dagflow.node_deco import NodeClass
from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.input_extra import MissingInputAddOne
import numpy as N

# Node functions
@NodeClass(output='array')
def Array(node, inputs, output):
    """Creates a note with single data output with predefined array"""
    outputs[0].data = N.arange(5, dtype='d')

@NodeClass(missing_input_handler=MissingInputAddOne(output_fmt='result'))
def Adder(node, inputs, output):
    """Adds all the inputs together"""
    out = None
    for input in inputs:
        if out is None:
            out=outputs[0].data = input.data.copy()
        else:
            out+=input.data

@NodeClass(missing_input_handler=MissingInputAddOne(output_fmt='result'))
def Multiplier(node, inputs, output):
    """Multiplies all the inputs together"""
    out = None
    for input in inputs:
        if out is None:
            out = outputs[0].data = input.data.copy()
        else:
            out*=input.data

# The actual code
with Graph() as graph:
    (in1, in2, in3, in4) = [Array(name) for name in ['n1', 'n2', 'n3', 'n4']]
    s = Adder('add')
    m = Multiplier('mul')

(in1, in2, in3) >> s
(in4, s) >> m

print('Result is:', m.outputs["result"].data)
savegraph(graph, 'output/dagflow_example.png')
```

The code produces the following graph:

![Image](example/dagflow_example.png)

For `n=[1, 2, 3, 4]` the output is:
```
Result is: [ 0.  3. 12. 27. 48.]
```
