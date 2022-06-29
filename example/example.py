#!/usr/bin/env python

from __future__ import print_function

from numpy import arange
from numpy.random import randint

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib import Product, Sum, makeArray
from dagflow.node_deco import NodeClass

Array = makeArray(arange(3, dtype="d"))

# The actual code
with Graph() as graph:
    (in1, in2, in3, in4) = (Array(name) for name in {"n1", "n2", "n3", "n4"})
    s = Sum("sum")
    m = Product("product")

    (in1, in2, in3) >> s
    (in4, s) >> m

    print("Result:", m.outputs.result.data)
    savegraph(graph, "dagflow_example_1.png")

# The actual code
with Graph() as graph:
    (in1, in2, in3, in4) = (Array(name) for name in {"n1", "n2", "n3", "n4"})
    s = Sum("sum1")
    s2 = Sum("sum2")
    m = Product("product")

    (in1, in2) >> s
    (in3, in4) >> s2
    (s, s2) >> m

    print("Result:", m.outputs.result.data)
    savegraph(graph, "dagflow_example_2.png")
