from .node import Node
from .tools import undefined
from .exception import CriticalError

class FunctionNode(Node):
    """Function signature: fcn(node, inputs, outputs)

    Note: _fcn should be a static function with signature (node, inputs, outputs)

    - Function defined as instance property will become a static method:
        class Node(...):
            def __init__(self):
                self._fcn = ...
        node = Node()
        node.fcn() # will have NO self provided as first argument

    - Fucntion defined in a nested class with staticmethod:
        class Other(Node
            @staticmethod
            def _fcn():
                ...

        node = Node()
        node.fcn() # will have NO self provided as first argument

    - [deprecated] Function defined as class property will become a bound method:
        class Node(...):
            _fcn = ...
        node = Node()
        node.fcn() # will have self provided as first argument

    - [deprecated] Function defined via staticmethod decorator as class property will become a static method:
        class Node(...):
            _fcn = staticmethod(...)
        node = Node()
        node.fcn() # will have NO self provided as first argument
    """

    def _stash_fcn(self):
        self._fcn_chain.append(self._fcn)
        return self._fcn

    def _make_wrap(self, prev_fcn, wrap_fcn):
        def wrapped_fcn(node, inputs, outputs):
            wrap_fcn(prev_fcn, node, inputs, outputs)

        return wrapped_fcn

    def _eval(self):
        return self._fcn(self, self.inputs, self.outputs)


class StaticNode(Node):
    """Function signature: fcn()"""

    _touch_inputs = True

    def __init__(self, *args, **kwargs):
        self._touch_inputs = kwargs.pop("touch_inputs", True)
        super().__init__(*args, **kwargs)

    def _eval(self):
        self._evaluated = True
        if self._touch_inputs:
            self.inputs.touch()
        ret = self._fcn()
        self._evaluated = False
        return ret

    def _stash_fcn(self):
        prev_fcn = self._fcn
        self._fcn_chain.append(prev_fcn)
        return lambda node, inputs, outputs: prev_fcn()

    def _make_wrap(self, prev_fcn, wrap_fcn):
        def wrapped_fcn():
            wrap_fcn(prev_fcn, self, self.inputs, self.outputs)

        return wrapped_fcn
