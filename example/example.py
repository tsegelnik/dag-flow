from __future__ import print_function

from dagflow.exception import CriticalError
from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.input_extra import MissingInputAddEach
from dagflow.lib import Array, Product, Sum, WeightedSum
from dagflow.node import FunctionNode
from numpy import arange, asarray
from numpy.random import randint

array = arange(3, dtype="d")

# Check predefined Array, Sum and Product
with Graph() as graph:
    (in1, in2, in3, in4) = (
        Array(name, array) for name in {"n1", "n2", "n3", "n4"}
    )
    s = Sum("sum")
    m = Product("product")

    (in1, in2, in3) >> s
    (in4, s) >> m
    m.close()

    print("Result:", m.outputs.result.data)
    savegraph(graph, "dagflow_example_1a.png")

# Check random generated Array, Sum and Product
with Graph() as graph:
    (in1, in2, in3, in4) = (
        Array(name, array) for name in {"n1", "n2", "n3", "n4"}
    )
    s = Sum("sum")
    m = Product("product")

    (in1, in2, in3) >> s
    (in4, s) >> m
    m.close()

    print("Result:", m.outputs.result.data)
    savegraph(graph, "dagflow_example_1b.png")

# Check predefined Array, two Sum's and Product
with Graph() as graph:
    (in1, in2, in3, in4) = (
        Array(name, array) for name in {"n1", "n2", "n3", "n4"}
    )
    s = Sum("sum")
    s2 = Sum("sum")
    m = Product("product")

    (in1, in2) >> s
    (in3, in4) >> s2
    (s, s2) >> m
    m.close()

    print("Result:", m.outputs.result.data)
    savegraph(graph, "dagflow_example_2.png")

# Check predefined Array, Sum, WeightedSum and Product
with Graph() as graph:
    (in1, in2, in3, in4) = (
        Array(name, array) for name in {"n1", "n2", "n3", "n4"}
    )
    weight = Array("weight", (2, 3))
    # The same result with other weight
    # weight = makeArray(5)("weight")
    s = Sum("sum")
    ws = WeightedSum("weightedsum")
    m = Product("product")

    (in1, in2) >> s
    (in3, in4) >> ws
    weight >> ws("weight")
    # TODO: check this issue
    # The way below does not work, due to automatic input naming
    # (in3, in4, weight) >> ws
    (s, ws) >> m
    m.close()

    print("Result:", m.outputs.result.data)
    savegraph(graph, "dagflow_example_3.png")


class ThreeInputsOneOutput(FunctionNode):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("missing_input_handler", MissingInputAddEach())
        super().__init__(*args, **kwargs)

    def check_input(self, name, iinput=None):
        super().check_input(name, iinput)

    def check_eval(self):
        super().check_eval()
        return True

    def _fcn(self, _, inputs, outputs):
        for i, output in enumerate(outputs):
            n = 3 * i
            out = output.data = inputs[n].data.copy()
            for input in inputs[n + 1 : 2 * n]:
                out += input.data

    @property
    def result(self):
        return [out.data for out in self.outputs]


with Graph() as graph:
    (in1, in2, in3) = (Array(name, array) for name in {"n1", "n2", "n3"})
    (in4, in5, in6) = (
        Array(name, asarray((1, 0, 0))) for name in {"n4", "n5", "n6"}
    )
    (in7, in8, in9) = (
        Array(name, asarray((3, 3, 3))) for name in {"n7", "n8", "n9"}
    )
    s = ThreeInputsOneOutput("3to1")
    (in1, in2, in3) >> s
    (in4, in5, in6) >> s
    (in7, in8, in9) >> s
    s.close()

    print("Result:", s.result)
    savegraph(graph, "dagflow_example_4.png")
