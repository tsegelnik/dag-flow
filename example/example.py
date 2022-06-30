#!/usr/bin/env python

from __future__ import print_function

from numpy import arange
from numpy.random import randint

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib import Product, Sum, makeArray, WeightedSum

Array = makeArray(arange(3, dtype="d"))

# Check predefined Array, Sum and Product
with Graph() as graph:
    (in1, in2, in3, in4) = (Array(name) for name in {"n1", "n2", "n3", "n4"})
    s = Sum("sum")
    m = Product("product")

    (in1, in2, in3) >> s
    (in4, s) >> m

    print("Result:", m.outputs.result.data)
    savegraph(graph, "dagflow_example_1a.png")

# Check random generated Array, Sum and Product
with Graph() as graph:
    (in1, in2, in3, in4) = (
        makeArray(randint(3, size=3))(name)
        for name in {"n1", "n2", "n3", "n4"}
    )
    s = Sum("sum")
    m = Product("product")

    (in1, in2, in3) >> s
    (in4, s) >> m

    print("Result:", m.outputs.result.data)
    savegraph(graph, "dagflow_example_1b.png")

# Check predefined Array, two Sum's and Product
with Graph() as graph:
    (in1, in2, in3, in4) = (Array(name) for name in {"n1", "n2", "n3", "n4"})
    s = Sum("sum")
    s2 = Sum("sum")
    m = Product("product")

    (in1, in2) >> s
    (in3, in4) >> s2
    (s, s2) >> m

    print("Result:", m.outputs.result.data)
    savegraph(graph, "dagflow_example_2.png")

# Check predefined Array, Sum, WeightedSum and Product
with Graph() as graph:
    (in1, in2, in3, in4) = (Array(name) for name in {"n1", "n2", "n3", "n4"})
    weight = makeArray((2, 3))("weight")
    s = Sum("sum")
    ws = WeightedSum("weightedsum")
    m = Product("product")

    (in1, in2) >> s
    (in3, in4) >> ws
    weight >> ws("weight")
    (s, ws) >> m

    print("Result:", m.outputs.result.data)
    savegraph(graph, "dagflow_example_3.png")
