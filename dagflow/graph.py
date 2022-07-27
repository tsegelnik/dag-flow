from __future__ import print_function

from .node_group import NodeGroup
from .tools import undefined


class Graph(NodeGroup):
    _context_graph = undefined("graph")
    _label = undefined("label")
    _name = "graph"
    _closed: bool = False
    _debug: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self._label = kwargs.pop("label", undefined("label"))
        self._name = kwargs.pop("name", "graph")
        self._debug = kwargs.pop("debug", False)
        if kwargs:
            raise RuntimeError(f"Unparsed arguments: {kwargs}!")

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def debug(self) -> bool:
        return self._debug

    @property
    def name(self) -> str:
        return self._name

    def add_node(self, name, **kwargs):
        from .node import FunctionNode
        return kwargs.pop("nodeclass", FunctionNode)(name, graph=self, **kwargs)

    def label(self, *args, **kwargs):
        if self._label:
            return self._label.format(self._label, nodes=len(self._nodes))

    def add_nodes(self, pairs, **kwargs):
        return (self.add_node(name, fcn, **kwargs) for name, fcn in pairs)

    def _add_input(self, input):
        # TODO: is it necessary?
        # self._inputs.append(input)
        pass

    def _add_output(self, output):
        # TODO: is it necessary?
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
        Graph._context_graph = undefined("graph")

    def close(self) -> bool:
        if self.debug:
            print(f"DEBUG: Graph '{self.name}': Closing...")
        if self._closed:
            return True
        self._closed = all(node.close() for node in self._nodes)
        if not self._closed:
            print(
                f"WARNING: Graph '{self.name}': Some nodes are still open: "
                f"'{tuple(node.name for node in self._nodes if not node.closed)}'!"
            )
        return self._closed

    def open(self) -> bool:
        if self.debug:
            print(f"DEBUG: Graph '{self.name}': Opening...")
        if not self._closed:
            return True
        self._closed = not all(node.open() for node in self._nodes)
        if self._closed:
            print(
                f"WARNING: Graph '{self.name}': Some nodes are still open: "
                f"'{tuple(node.name for node in self._nodes if node.closed)}'!"
            )
        return not self._closed
