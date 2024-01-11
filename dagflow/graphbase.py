from .shift import rshift


class GraphBase:
    __slots__ = ("_nodes", "_nodes_set")
    _nodes: list
    _nodes_set: set

    def __init__(self, *args):
        self._nodes = list(args)
        self._nodes_set = set(args)

        if len(self._nodes_set) != len(self._nodes):
            raise RuntimeError("There are duplicated nodes")

    def register_node(self, node):
        if node in self._nodes_set:
            return
        self._nodes.append(node)
        self._nodes_set.add(node)

    def _wrap_fcns(self, *args):
        for node in self._nodes:
            node._wrap_fcn(*args)

    def _unwrap_fcns(self):
        for node in self._nodes:
            node._unwrap_fcn()

    def print(self):
        print(f"Graph consists of {len(self._nodes)} nodes:")
        for node in self._nodes:
            node.print()

    def __rrshift__(self, other):
        """
        other >> self
        """
        return rshift(other, self)

    def __iter__(self):
        """
        iterate inputs

        To be used with >>/<< operators which take only disconnected inputs
        """
        return iter(self._nodes)
