from typing import Any, Callable, Dict

from .exception import DagflowError
from .node import Node


class FunctionNode(Node):
    """Function signature: fcn(node)

    - Function defined as instance property will become a static method:
        class Node(...):
            def __init__(self):
                self._fcn = ...
        node = Node()
        node.fcn() # will have NO self provided as first argument

    - Fucntion defined in a nested class with staticmethod:
        class Other(Node)
            def _fcn():
                ...

        node = Node()
        node.fcn() # will have NO self provided as first argument
    """

    __slots__ = ("fcn", "_fcn_chain", "_functions")

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self._fcn_chain = []

        self._functions: Dict[Any, Callable] = {"default": self._fcn}
        self.fcn = self._functions["default"]

    def _stash_fcn(self):
        self._fcn_chain.append(self.fcn)
        return self.fcn

    def _make_wrap(self, prev_fcn, wrap_fcn):
        def wrapped_fcn():
            wrap_fcn(prev_fcn, self)

        return wrapped_fcn

    def _wrap_fcn(self, wrap_fcn, *other_fcns):
        prev_fcn = self._stash_fcn()
        self.fcn = self._make_wrap(prev_fcn, wrap_fcn)
        if other_fcns:
            self._wrap_fcn(*other_fcns)

    def _unwrap_fcn(self):
        if not self._fcn_chain:
            raise DagflowError("Unable to unwrap bare function")
        self.fcn = self._fcn_chain.pop()

    def _fcn(self):
        pass

    def _eval(self):
        return self.fcn()
