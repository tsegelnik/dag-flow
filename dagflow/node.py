from __future__ import print_function

from .exception import CriticalError
from .graph import Graph
from .input import Input
from .legs import Legs
from .logger import Logger
from .output import Output
from .shift import lshift, rshift
from .tools import IsIterable, undefined


class Node(Legs):
    _name = undefined("name")
    _label = undefined("label")
    _graph = undefined("graph")
    _fcn = undefined("function")
    _fcn_chain = None

    # Taintflag and status
    _tainted: bool = True
    _frozen: bool = False
    _frozen_tainted: bool = False
    _invalid: bool = False
    _closed: bool = False
    _allocated: bool = False
    _evaluated: bool = False

    # Options
    _debug: bool = False
    _auto_freeze: bool = False
    _immediate: bool = False
    # _always_tainted: bool = False

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
        self._debug = kwargs.pop(
            "debug", self.graph.debug if self.graph else False
        )
        self._label = kwargs.pop("label", undefined("label"))
        if (logger := kwargs.pop("logger", False)) or (
            self.graph and (logger := self.graph.logger)
        ):
            self._logger = logger
        else:
            from .logger import get_logger

            self._logger = get_logger(
                filename=kwargs.pop("logfile", None),
                debug=self.debug,
                console=kwargs.pop("console", True),
                formatstr=kwargs.pop("logformat", None),
                name=kwargs.pop("loggername", None),
            )
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
    def logger(self) -> Logger:
        return self._logger

    @property
    def tainted(self) -> bool:
        return self._tainted

    @property
    def frozen_tainted(self) -> bool:
        return self._frozen_tainted

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def auto_freeze(self) -> bool:
        return self._auto_freeze

    # @property
    # def always_tainted(self) -> bool:
    # return self._always_tainted

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def debug(self) -> bool:
        return self._debug

    @property
    def evaluated(self) -> bool:
        return self._evaluated

    @property
    def immediate(self) -> bool:
        return self._immediate

    @property
    def invalid(self) -> bool:
        return self._invalid

    @invalid.setter
    def invalid(self, invalid) -> None:
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

    def __call__(self, name, iinput=undefined("iinput")):
        for inp in self.inputs:
            if inp.name == name:
                return inp
        if not self.closed:
            return self._add_input(name, iinput)
        self.logger.warning(
            f"Node '{self.name}': Input '{name}' doesn't exist, "
            "and a modification of the closed node is restricted!"
        )

    def label(self, *args, **kwargs):
        if self._label:
            kwargs.setdefault("name", self._name)
            return self._label.format(*args, **kwargs)
        return self._label

    def allocate(self, **kwargs):
        if self._allocated:
            self.logger.warning(
                f"Node '{self.name}': The memory is already allocated!"
            )
            return self._allocated
        self.logger.debug(f"Node '{self.name}': Allocate the memory...")
        try:
            self._allocated = all(
                out.allocate(**kwargs) for out in self.outputs
            )
        except Exception:
            self._allocated = False
        return self._allocated

    def add_input(self, name, iinput=undefined("iinput")):
        if not self.closed:
            return self._add_input(name, iinput)
        self.logger.warning(
            f"Node '{self.name}': "
            "A modification of the closed node is restricted!"
        )

    def _add_input(self, name, iinput=undefined("iinput")):
        if IsIterable(name):
            return tuple(self._add_input(n) for n in name)
        if name in self.inputs:
            raise ValueError(f"Input {self.name}.{name} already exist!")
        inp = Input(name, self, iinput)
        self.inputs += inp
        if self._graph:
            self._graph._add_input(inp)
        return inp

    def add_output(self, name):
        if not self.closed:
            return self._add_output(name)
        self.logger.warning(
            f"Node '{self.name}': "
            "A modification of the closed node is restricted!"
        )

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

    def add_pair(self, iname, oname):
        if not self.closed:
            return self._add_pair(iname, oname)
        self.logger.warning(
            f"Node '{self.name}': "
            "A modification of the closed node is restricted!"
        )

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
        self.logger.debug(f"Node '{self.name}': Evaluating node...")
        if self.invalid:
            raise CriticalError("Unable to evaluate invalid transformation!")
        if not self._closed:
            raise CriticalError("Close the node before evaluation!")
        # if not self._allocated:
        #    raise CriticalError("Allocate the memory before evaluation!")
        self._evaluated = True
        try:
            ret = self._eval()
        except Exception as exc:
            self._evaluated = False
            raise exc from RuntimeError(
                "An exception occured during evaluation!"
            )
        self._evaluated = False
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

    def close(self, **kwargs) -> bool:
        self.logger.debug(f"Node '{self.name}': Closing...")
        if self._closed:
            return self._closed
        self._closed = all(inp.close(**kwargs) for inp in self.inputs)
        if not self._closed:
            self.logger.warning(
                f"Node '{self.name}': Some inputs are still open: "
                f"'{tuple(inp.name for inp in self.inputs if not inp.closed)}'!"
            )
        else:
            self._closed = all(out.close(**kwargs) for out in self.outputs)
            if not self._closed:
                self.logger.warning(
                    f"Node '{self.name}': Some outputs are still open: "
                    f"'{tuple(out.name for out in self.outputs if not out.closed)}'!"
                )
                return False
            # self.allocate(**kwargs)
        return self._closed

    def open(self) -> bool:
        self.logger.debug(f"Node '{self.name}': Opening...")
        if not self._closed:
            return True
        self._closed = not all(inp.open() for inp in self.inputs)
        if self._closed:
            self.logger.warning(
                f"Node '{self.name}': Some inputs are still closed: "
                f"'{tuple(inp.name for inp in self.inputs if inp.closed)}'!"
            )
        else:
            self._closed = not all(out.open() for out in self.outputs)
            if self._closed:
                self.logger.warning(
                    f"Node '{self.name}': Some outputs are still closed: "
                    f"'{tuple(out.name for out in self.outputs if out.closed)}'!"
                )
        return self._closed


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
        self._evaluated = True
        try:
            ret = self._fcn(self, self.inputs, self.outputs)
        except Exception as exc:
            self._evaluated = False
            raise exc
        self._evaluated = False
        return ret

    def eval(self):
        self.logger.debug(f"Node '{self.name}': Evaluating...")
        if not self._closed:
            raise CriticalError(
                "Close the node before evaluation! Unclosed inputs:"
                f"'{tuple(inp.name for inp in self.inputs if not inp.closed)}',"
                " Unclosed outputs: "
                f"'{tuple(out.name for out in self.outputs if not out.closed)}'"
            )
        # if not self._allocated:
        #    raise CriticalError(
        #        "Memory is not allocated! Problem inputs:"
        #        f"'{tuple(inp.name for inp in self.inputs if not inp.allocated)}',"
        #        " Problem outputs: "
        #        f"'{tuple(out.name for out in self.outputs if not out.allocated)}'"
        #    )
        return self._eval()

    def add_input(self, name, iinput=undefined("iinput")):
        if self.closed:
            self.logger.warning(
                f"Node '{self.name}': "
                "A modification of the closed node is restricted!"
            )
        else:
            return self._add_input(name, iinput)

    def _add_input(self, name, iinput=undefined("iinput")):
        try:
            self.check_input(name, iinput)
        except CriticalError as exc:
            raise exc from CriticalError(
                f"Cannot add the input ({name=}, {iinput=}) due to "
                "critical error!"
            )
        except Exception as exc:
            print(exc)
        return super()._add_input(name, iinput)

    def _check_input(self) -> bool:
        return True

    def _check_eval(self) -> bool:
        return True

    def check_input(self, name=None, iinput=None) -> bool:
        """Checks a signature of the function at the input connection stage"""
        self.logger.debug(
            f"Node '{self.name}': "
            f"Checking a possibility to add new input '{name}'..."
        )
        return self._check_input()

    def check_eval(self) -> bool:
        """Checks a signature of the function at the evaluation stage"""
        self.logger.debug(
            f"Node '{self.name}': Checking the evaluation access..."
        )
        return self._check_eval()

    def _close(self) -> bool:
        super()._close()
        if self._closed:
            self._closed = self.check_eval()
        return self._closed


class StaticNode(Node):
    """Function signature: fcn()"""

    _touch_inputs = True

    def __init__(self, *args, **kwargs):
        self._touch_inputs = kwargs.pop("touch_inputs", True)
        super().__init__(*args, **kwargs)

    def _eval(self):
        self._evaluated = True
        if self._touch_inputs:
            self.inputs._touch()
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
