from __future__ import print_function

from .graph import Graph
from .input import Input
from .legs import Legs
from .output import Output
from .shift import lshift, rshift
from .tools import (
    IsIterable,
    undefinedfunction,
    undefinedgraph,
    undefinedname,
    undefinedoutput,
)


class Node(Legs):
    _name = undefinedname
    _label = undefinedname
    _graph = undefinedgraph
    _fcn = undefinedfunction
    _fcn_chain = None

    # Taintflag and status
    _tainted = True
    _frozen = False
    _frozen_tainted = False
    _invalid = False

    _evaluating = False

    # Options
    _auto_freeze = False
    _immediate = False
    # _always_tainted = False

    def __init__(self, name, **kwargs):
        super().__init__(
            missing_input_handler=kwargs.pop("missing_input_handler", None),
        )
        self._name = name
        if newfcn := kwargs.pop("fcn", None):
            self._fcn = newfcn
        self._fcn_chain = []
        self.graph = kwargs.pop("graph", None)
        if not self.graph:
            self.graph = Graph.current()
        self._label = kwargs.pop("label", undefinedname)
        for opt in {"immediate", "auto_freeze", "frozen"}:
            if value := kwargs.pop(opt, None):
                setattr(self, f"_{opt}", bool(value))
        if input := kwargs.pop("input", None):
            self._add_input(input)
        if output := kwargs.pop("output", None):
            self._add_output(output)
        if kwargs:
            raise ValueError(f"Unparsed arguments: {kwargs}!")

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def tainted(self):
        return self._tainted

    @property
    def frozen_tainted(self):
        return self._frozen_tainted

    @property
    def frozen(self):
        return self._frozen

    @property
    def auto_freeze(self):
        return self._auto_freeze

    # @property
    # def always_tainted(self):
    # return self._always_tainted

    @property
    def evaluating(self):
        return self._evaluating

    @property
    def immediate(self):
        return self._immediate

    @property
    def invalid(self):
        return self._invalid

    @invalid.setter
    def invalid(self, invalid):
        if invalid:
            self._tainted = True
            self._frozen = False
            self._frozen_tainted = False
        else:
            for input in self.inputs:
                if input.invalid:
                    return
        self._invalid = invalid
        for output in self.outputs:
            output.invalid = invalid

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, graph):
        if not graph:
            return
        if self._graph:
            raise ValueError("Graph is already defined")
        self._graph = graph
        self._graph.register_node(self)

    def __call__(self, name, iinput=undefinedoutput):
        for inp in self.inputs:
            if inp.name == name:
                return inp
        return self._add_input(name, iinput)

    def label(self, *args, **kwargs):
        if self._label:
            kwargs.setdefault("name", self._name)
            return self._label.format(*args, **kwargs)
        return self._label
    
    def _add_input(self, name, iinput=undefinedoutput):
        if IsIterable(name):
            return tuple(self._add_input(n) for n in name)
        if name in self.inputs:
            raise ValueError(f"Input {self.name}.{name} already exist!")
        inp = Input(name, self, iinput)
        self.inputs += inp
        if self._graph:
            self._graph._add_input(inp)
        return inp

    def _add_output(self, name):
        if IsIterable(name):
            return tuple(self._add_output(n) for n in name)
        if isinstance(name, Output):
            if name.name in self.outputs:
                raise RuntimeError(
                    f"Output {self.name}.{name.name} already exist!"
                )
            if name.node:
                raise RuntimeError(
                    f"Output {name.name} is connected to another node {self.name}!"
                )
            name._node = self
            self.outputs += name
            if self._graph:
                self._graph._add_output(name)
            return name
        if name in self.outputs:
            raise RuntimeError(f"Output {self.name}.{name} already exist!")
        output = Output(name, self)
        self.outputs += output
        if self._graph:
            self._graph._add_output(output)
        return output

    def _add_pair(self, iname, oname):
        output = self._add_output(oname)
        return self._add_input(iname, output), output

    def _wrap_fcn(self, wrap_fcn, *other_fcns):
        prev_fcn = self._stash_fcn()
        self._fcn = self._make_wrap(prev_fcn, wrap_fcn)
        if other_fcns:
            self._wrap_fcn(*other_fcns)

    def _unwrap_fcn(self):
        if not self._fcn_chain:
            raise RuntimeError("Unable to unwrap bare function")
        self._fcn = self._fcn_chain.pop()

    def _stash_fcn(self):
        raise RuntimeError(
            "Unimplemented method: use FunctionNode, StaticNode or MemberNode"
        )

    def _make_wrap(self, prev_fcn, wrap_fcn):
        raise RuntimeError(
            "Unimplemented method: use FunctionNode, StaticNode or MemberNode"
        )

    def touch(self, force=False):
        if self._frozen:
            return
        if not self._tainted and not force:
            return
        ret = self.eval()
        self._tainted = False  # self._always_tainted
        if self._auto_freeze:
            self._frozen = True
        return ret

    def _eval(self):
        raise RuntimeError(
            "Unimplemented method: use FunctionNode, StaticNode or MemberNode"
        )

    def eval(self):
        if self.invalid:
            raise RuntimeError("Unable to evaluate invalid transformation")
        self._evaluating = True
        try:
            ret = self._eval()
        except Exception as exc:
            self._evaluating = False
            raise exc
        self._evaluating = False
        return ret

    def freeze(self):
        if self._frozen:
            return
        if self._tainted:
            raise RuntimeError("Unable to freeze tainted node")
        self._frozen = True
        self._frozen_tainted = False

    def unfreeze(self):
        if not self._frozen:
            return
        self._frozen = False
        if self._frozen_tainted:
            self._frozen_tainted = False
            self.taint(force=True)

    def taint(self, force=False):
        if self._tainted and not force:
            return
        if self._frozen:
            self._frozen_tainted = True
            return
        self._tainted = True
        ret = self.touch() if self._immediate else None
        for output in self.outputs:
            output.taint(force)
        return ret

    def print(self):
        print(
            f"Node {self._name}: ->[{len(self.inputs)}],[{len(self.outputs)}]->"
        )
        for i, input in enumerate(self.inputs):
            print("  ", i, input)
        for i, output in enumerate(self.outputs):
            print("  ", i, output)


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _stash_fcn(self):
        self._fcn_chain.append(self._fcn)
        return self._fcn

    def _make_wrap(self, prev_fcn, wrap_fcn):
        def wrapped_fcn(node, inputs, outputs):
            wrap_fcn(prev_fcn, node, inputs, outputs)
        return wrapped_fcn

    def _eval(self):
        self._evaluating = True
        try:
            ret = self._fcn(self, self.inputs, self.outputs)
        except Exception as exc:
            self._evaluating = False
            raise exc
        self._evaluating = False
        return ret


class StaticNode(Node):
    """Function signature: fcn()"""

    _touch_inputs = True

    def __init__(self, *args, **kwargs):
        self._touch_inputs = kwargs.pop("touch_inputs", True)
        super().__init__(*args, **kwargs)

    def _eval(self):
        self._evaluating = True
        if self._touch_inputs:
            self.inputs._touch()
        ret = self._fcn()
        self._evaluating = False
        return ret

    def _stash_fcn(self):
        prev_fcn = self._fcn
        self._fcn_chain.append(prev_fcn)
        return lambda node, inputs, outputs: prev_fcn()

    def _make_wrap(self, prev_fcn, wrap_fcn):
        def wrapped_fcn():
            wrap_fcn(prev_fcn, self, self.inputs, self.outputs)
        return wrapped_fcn
