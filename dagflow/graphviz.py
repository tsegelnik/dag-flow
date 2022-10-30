from __future__ import print_function

from collections import OrderedDict

from .input import Input
from .output import Output
from .printl import printl

try:
    import pygraphviz as G
except ImportError:
    GraphDot = None
    savegraph = None
else:

    def savegraph(graph, *args, **kwargs):
        gd = GraphDot(graph, **kwargs)
        gd.savegraph(*args)

    class GraphDot:
        _graph = None

        def __init__(self, dag, **kwargs):
            kwargs.setdefault("fontsize", 10)
            kwargs.setdefault("labelfontsize", 10)
            kwargs.setdefault("rankdir", "LR")
            self._nodes = OrderedDict()
            self._nodes_open_input = OrderedDict()
            self._nodes_open_output = OrderedDict()
            self._edges = OrderedDict()
            self._graph = G.AGraph(directed=True, strict=False, **kwargs)
            if label := kwargs.pop("label", dag.label()):
                self.set_label(label)
            self._transform(dag)

        def _transform(self, dag):
            for nodedag in dag._nodes:
                self._add_node(nodedag)
            for nodedag in dag._nodes:
                self._add_open_inputs(nodedag)
                self._add_edges(nodedag)
            self.update_style()

        def get_id(self, object, suffix=""):
            return f"{type(object).__name__}_{id(object)}{suffix}"

        def _add_node(self, nodedag):
            styledict = {"shape": "Mrecord"}
            target = self.get_id(nodedag)
            self._graph.add_node(
                target, label=nodedag.label() or nodedag.name, **styledict
            )
            nodedot = self._graph.get_node(target)
            self._nodes[nodedag] = nodedot

        def _add_open_inputs(self, nodedag):
            for input in nodedag.inputs:
                if not input.connected():
                    self._add_open_input(input, nodedag)

        def _add_open_input(self, input, nodedag):
            styledict = {}
            source = self.get_id(input, "_in")
            target = self.get_id(nodedag)

            self._graph.add_node(source, shape="point", **styledict)
            self._graph.add_edge(source, target, **styledict)

            nodein = self._graph.get_node(source)
            edge = self._graph.get_edge(source, target)
            nodeout = self._graph.get_node(target)

            self._nodes_open_input[input] = nodein
            self._edges[input] = (nodein, edge, nodeout)

        def _add_edges(self, nodedag):
            for output in nodedag.outputs:
                if output.connected():
                    for input in output.inputs:
                        self._add_edge(nodedag, output, input)
                else:
                    self._add_open_output(nodedag, output)

        def _add_open_output(self, nodedag, output):
            styledict = {}
            source = self.get_id(nodedag)
            target = self.get_id(output, "_out")

            self._graph.add_node(target, shape="point", **styledict)
            self._graph.add_edge(
                source, target, arrowhead="empty", **styledict
            )
            nodein = self._graph.get_node(source)
            edge = self._graph.get_edge(source, target)
            nodeout = self._graph.get_node(target)

            self._nodes_open_output[output] = nodeout
            self._edges[output] = (nodein, edge, nodeout)

        def _add_edge(self, nodedag, output, input):
            styledict = {}

            source = self.get_id(nodedag)
            target = self.get_id(input.node)
            self._graph.add_edge(source, target, **styledict)

            nodein = self._graph.get_node(source)
            edge = self._graph.get_edge(source, target)
            nodeout = self._graph.get_node(target)

            self._edges[input] = (nodein, edge, nodeout)

        def _set_style_node(self, node, attr):
            if not node:
                attr["color"] = "gray"
            elif node.invalid:
                attr["color"] = "black"
            elif node.evaluated:
                attr["color"] = "gold"
            elif node.tainted:
                attr["color"] = "red"
            elif node.frozen_tainted:
                attr["color"] = "blue"
            elif node.frozen:
                attr["color"] = "cyan"
            elif node.immediate:
                attr["color"] = "green"
            else:
                attr["color"] = "forestgreen"

        def _set_style_edge(self, obj, attrin, attr, attrout):
            if isinstance(obj, Input):
                if obj.connected():
                    node = obj.output.node
                else:
                    node = None
                    self._set_style_node(node, attrin)
            else:
                node = obj.node
                self._set_style_node(node, attrout)

            self._set_style_node(node, attr)

            if node:
                if node.frozen:
                    attrin["style"] = "dashed"
                    attr["style"] = "dashed"
                    # attr['arrowhead']='tee'
                else:
                    attr["style"] = ""

        def update_style(self):
            for nodedag, nodedot in self._nodes.items():
                self._set_style_node(nodedag, nodedot.attr)

            for object, (nodein, edge, nodeout) in self._edges.items():
                self._set_style_edge(
                    object, nodein.attr, edge.attr, nodeout.attr
                )

        def set_label(self, label):
            self._graph.graph_attr["label"] = label

        def savegraph(self, fname, verbose=True):
            if verbose:
                printl("Write output file:", fname)

            if fname.endswith(".dot"):
                self._graph.write(fname)
            else:
                self._graph.layout(prog="dot")
                self._graph.draw(fname)
