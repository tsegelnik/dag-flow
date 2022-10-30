from __future__ import print_function

from .input_extra import MissingInputAddOne
from .membernode import MemberNode, StaticMemberNode
from .node import FunctionNode, StaticNode


def NodeClass(fcn=None, **kwargsdeco):
    """
    Create a node class based on a function. The result is a class
    which should be instantiated in order to be used as node

    NOTE: The decorator works only for one instance of class!
    """
    if not fcn:
        return lambda fcn1: NodeClass(fcn1, **kwargsdeco)
    kwargsdeco["fcn"] = fcn
    parent = kwargsdeco.pop("parent", FunctionNode) 
    class NewNodeClass(parent):
        def __init__(self, *args, **kwargsclass):
            self._fcn = fcn
            kwargs = dict(kwargsclass, **kwargsdeco)
            parent.__init__(self, *args, **kwargs)
    return NewNodeClass


def _NodeInstance(fcn=None, **kwargsinstance):
    """
    Create a node class instantiate it.
    The result is a node class instance which may be used as node.
    The node type is defined by the parent keyword
    """
    if not fcn:
        return lambda fcn1: _NodeInstance(fcn1, **kwargsinstance)
    kwargsclass = kwargsinstance.pop("class_kwargs", {})
    kwargsclass.setdefault("name", fcn.__name__)
    cls = NodeClass(fcn, **kwargsclass)
    return cls(**kwargsinstance)


def NodeInstance(fcn=None, **kwargsinstance):
    """
    Function signature _fcn(node, inputs, ouputs)
    """
    return _NodeInstance(fcn, **kwargsinstance)


def NodeInstanceStatic(fcn=None, **kwargsinstance):
    """
    Function signature _fcn()
    """
    kwargsinstance.setdefault("output", "result")
    kwargsinstance["class_kwargs"] = dict(
        parent=StaticNode, missing_input_handler=MissingInputAddOne()
    )
    return _NodeInstance(fcn, **kwargsinstance)


def NodeInstanceMember(fcn=None, **kwargsinstance):
    """
    Function signature _fcn(master, node, inputs, ouputs)
    """
    kwargsinstance["class_kwargs"] = dict(parent=MemberNode)
    return _NodeInstance(fcn, parent=MemberNode, **kwargsinstance)


def NodeInstanceStaticMember(fcn=None, **kwargsinstance):
    """
    Function signature _fcn(master)
    """
    kwargsinstance.setdefault("output", "result")
    kwargsinstance["class_kwargs"] = dict(
        parent=StaticMemberNode, missing_input_handler=MissingInputAddOne()
    )
    return _NodeInstance(fcn, **kwargsinstance)
