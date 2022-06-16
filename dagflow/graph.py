from __future__ import print_function
from .tools import undefinedgraph, undefinedname
from .node_group import NodeGroup


class Graph(NodeGroup):
    _context_graph = undefinedgraph
    _label = undefinedname

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self._label = kwargs.pop("label", undefinedname)

        if kwargs:
            raise RuntimeError("Unparsed arguments: {!s}".format(kwargs))

    def add_node(self, name, **kwargs):
        from .node import FunctionNode

        NodeClass = kwargs.pop("nodeclass", FunctionNode)
        return NodeClass(name, graph=self, **kwargs)

    def label(self, *args, **kwargs):
        if self._label:
            return self._label.format(self._label, nodes=len(self._nodes))

    def add_nodes(self, pairs, **kwargs):
        return (self.add_node(name, fcn, **kwargs) for name, fcn in pairs)

    def _add_input(self, input):
        # self._inputs.append(input)
        pass

    def _add_output(self, output):
        # self._outputs.append(output)
        pass

    def print(self):
        print(f"Graph with {len(self._nodes)} nodes")
        for node in self._nodes:
            node.print()

    @classmethod
    def current(cls):
        return cls._context_graph

    def __enter__(self):
        Graph._context_graph = self
        return self

    def __exit__(self, *args, **kwargs):
        Graph._context_graph = undefinedgraph
