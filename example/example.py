from __future__ import print_function

from numpy import arange, asarray, copyto, result_type
from numpy.random import randint

from dagflow.exception import CriticalError
from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.input_extra import MissingInputAddEach
from dagflow.lib import Array, Product, Sum, WeightedSum
from dagflow.node import FunctionNode

array = arange(3, dtype="d")
debug = True


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
            out = output.data
            copyto(out, inputs[3 * i].data)
            for input in inputs[3 * i + 1 : (i + 1) * 3]:
                out += input.data

    @property
    def result(self):
        return [out.data for out in self.outputs]

    def _shapefunc(self, node) -> None:
        """A output takes this function to determine the shape"""
        return node.inputs[0].data.shape

    def _typefunc(self, node) -> None:
        """A output takes this function to determine the dtype"""
        return result_type(*tuple(inp.dtype for inp in node.inputs))


# Check predefined Array, Sum and Product
with Graph(debug=debug) as graph:
    (in1, in2, in3, in4) = (
        Array(name, array) for name in ("n1", "n2", "n3", "n4")
    )
    s = Sum("sum")
    m = Product("product")

    (in1, in2, in3) >> s
    (in4, s) >> m
    m.close()

    print("Result:", m.outputs.result.data)
    savegraph(graph, "dagflow_example_1a.png")

# Check random generated Array, Sum and Product
with Graph(debug=debug) as graph:
    (in1, in2, in3, in4) = (
        Array(name, array) for name in ("n1", "n2", "n3", "n4")
    )
    s = Sum("sum")
    m = Product("product")

    (in1, in2, in3) >> s
    (in4, s) >> m
    m.close()

    print("Result:", m.outputs.result.data)
    savegraph(graph, "dagflow_example_1b.png")

# Check predefined Array, two Sum's and Product
with Graph(debug=debug) as graph:
    (in1, in2, in3, in4) = (
        Array(name, array) for name in ("n1", "n2", "n3", "n4")
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
with Graph(debug=debug) as graph:
    (in1, in2, in3, in4) = (
        Array(name, array) for name in ("n1", "n2", "n3", "n4")
    )
    weight = Array("weight", (2, 3))
    # The same result with other weight
    # weight = makeArray(5)("weight")
    s = Sum("sum")
    ws = WeightedSum("weightedsum")
    m = Product("product")

    (in1, in2) >> s  # [0,2,4]
    (in3, in4) >> ws
    {"weight": weight} >> ws  # [0,1,2] * 2 + [0,1,2] * 3 = [0,5,10]
    # NOTE: also it is possible to use the old style binding:
    #weight >> ws("weight")
    (s, ws) >> m  # [0,2,4] * [0,5,10] = [0,10,40]
    m.close()

    print("Result:", m.outputs.result.data)
    savegraph(graph, "dagflow_example_3.png")


with Graph(debug=debug) as graph:
    (in1, in2, in3) = (Array(name, array) for name in ("n1", "n2", "n3"))
    (in4, in5, in6) = (
        Array(name, (1, 0, 0)) for name in ("n4", "n5", "n6")
    )
    (in7, in8, in9) = (
        Array(name, (3, 3, 3)) for name in ("n7", "n8", "n9")
    )
    s = ThreeInputsOneOutput("3to1")
    (in1, in2, in3) >> s
    (in4, in5, in6) >> s
    (in7, in8, in9) >> s
    s.close()

    print("Result:", s.result)
    savegraph(graph, "dagflow_example_4.png")
