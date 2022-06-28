from __future__ import print_function
from dagflow import shift

class NodeGroup:
    _nodes  = None

    def __init__(self, *args):
        self._nodes   = list(args)

    def register_node(self, node):
        self._nodes.append(node)

    def _wrap_fcns(self, *args):
        for node in self._nodes:
            node._wrap_fcn(*args)

    def _unwrap_fcns(self):
        for node in self._nodes:
            node._unwrap_fcn()

    def print(self):
        print('Group of {} nodes:'.format(len(self._nodes)))
        for node in self._nodes:
            node.print()

    def __lshift__(self, other):
        """
        self << other
        """
        return shift.lshift(self, other)

    def __rrshift__(self, other):
        """
        other >> self
        """
        return shift.lshift(self, other)

    def __iter__(self):
        """
        iterate inputs

        To be used with >>/<< operators which take only disconnected inputs
        """
        return iter(self._nodes)

