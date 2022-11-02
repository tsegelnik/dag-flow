from .shift import lshift


class NodeGroup:
    _nodes: list = None

    def __init__(self, *args):
        self._nodes = list(args)

    def register_node(self, node):
        self._nodes.append(node)

    def _wrap_fcns(self, *args):
        for node in self._nodes:
            node._wrap_fcn(*args)

    def _unwrap_fcns(self):
        for node in self._nodes:
            node._unwrap_fcn()

    def print(self):
        print(f"Group of {len(self._nodes)} nodes:")
        for node in self._nodes:
            node.print()

    def __lshift__(self, other):
        """
        self << other
        """
        return lshift(self, other)

    def __rrshift__(self, other):
        """
        other >> self
        """
        return lshift(self, other)

    def __iter__(self):
        """
        iterate inputs

        To be used with >>/<< operators which take only disconnected inputs
        """
        return iter(self._nodes)
