
from .exception import CriticalError
from .input import Input
from .legs import Legs
from .logger import Logger
from .output import Output, SettableOutput
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
    _allocatable: bool = True
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
        if typefunc := kwargs.pop("typefunc", None):
            self._typefunc = typefunc
        self._fcn_chain = []
        self.graph = kwargs.pop("graph", None)
        if not self.graph:
            from .graph import Graph

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
        self._allocatable = kwargs.pop("allocatable", True)
        if input := kwargs.pop("input", None):
            self._add_input(input)
        if output := kwargs.pop("output", None):
            self._add_output(output)
        if kwargs:
            raise ValueError(f"Unparsed arguments: {kwargs}!")
        self.logger.debug(
            f"Node '{self.name}': The node is instantiated with following "
            f"options: allocatable={self._allocatable}, label={self._label}, "
            f"debug={self._debug}"
        )

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
    def allocated(self) -> bool:
        return self._allocated

    @property
    def allocatable(self) -> bool:
        return self._allocatable

    @property
    def immediate(self) -> bool:
        return self._immediate

    @property
    def invalid(self) -> bool:
        return self._invalid

    @invalid.setter
    def invalid(self, invalid) -> None:
        if invalid:
            self.invalidate_self()
        else:
            if any(input.invalid for input in self.inputs):
                    return
            self._invalid = False
        for output in self.outputs:
            output.invalid = invalid

    def invalidate_self(self) -> None:
        self._tainted = True
        self._frozen = False
        self._frozen_tainted = False
        self._invalid = True

    def invalidate_children(self) -> None:
        for output in self.outputs:
            output.invalid = True

    def invalidate_parents(self) -> None:
        for input in self.inputs:
            node = input.node
            node.invalidate_self()
            node.invalidate_parents()

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

    def __call__(self, name, parent_output=undefined("parent_output")):
        for inp in self.inputs:
            if inp.name == name:
                return inp
        if not self.closed:
            return self._add_input(name, parent_output)
        self.logger.warning(
            f"Node '{self.name}': Input '{name}' doesn't exist, "
            "and a modification of the closed node is restricted!"
        )

    def label(self, *args, **kwargs):
        if self._label:
            kwargs.setdefault("name", self._name)
            return self._label.format(*args, **kwargs)
        return self._label

    def add_input(self, name, parent_output=undefined("parent_output")):
        if not self.closed:
            return self._add_input(name, parent_output)
        self.logger.warning(
            f"Node '{self.name}': "
            "A modification of the closed node is restricted!"
        )

    def _add_input(self, name, parent_output=undefined("parent_output")):
        self.logger.debug(
            f"Node '{self.name}': Adding input '{name}' with parent_output='{parent_output}'..."
        )
        if IsIterable(name):
            return tuple(self._add_input(n) for n in name)
        if name in self.inputs:
            raise ValueError(f"Input {self.name}.{name} already exist!")
        inp = Input(name, self, parent_output)
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

    def _add_output(self, name, *, settable=False, **kwargs):
        if IsIterable(name):
            return tuple(self._add_output(n) for n in name)
        kwargs.setdefault("allocatable", self._allocatable)
        kwargs.setdefault("typefunc", self._typefunc)
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
        if settable:
            output = SettableOutput(name, self, **kwargs)
        else:
            output = Output(name, self, **kwargs)
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
        self.logger.info(f"Node '{self.name}': Evaluating node...")
        if self.invalid:
            raise CriticalError("Unable to evaluate invalid transformation!")
        if not self._closed:
            raise CriticalError(
                "Close the node before evaluation! Unclosed inputs:"
                f"'{tuple(inp.name for inp in self.inputs if not inp.closed)}',"
                " Unclosed outputs: "
                f"'{tuple(out.name for out in self.outputs if not out.closed)}'"
            )
        if not self._allocated:
            raise CriticalError(
                "Memory is not allocated! Problem inputs:"
                f"'{tuple(inp.name for inp in self.inputs if not inp.allocated)}',"
                " Problem outputs: "
                f"'{tuple(out.name for out in self.outputs if not out.allocated)}'"
            )
        self._evaluated = True
        try:
            ret = self._eval()
            self.logger.debug(f"Node '{self.name}': Evaluated return={ret}")
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
            output.taint_children(force)
        return ret

    def print(self):
        print(
            f"Node {self._name}: ->[{len(self.inputs)}],[{len(self.outputs)}]->"
        )
        for i, input in enumerate(self.inputs):
            print("  ", i, input)
        for i, output in enumerate(self.outputs):
            print("  ", i, output)

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        raise RuntimeError(
            "Unimplemented method: the method must be overridden!"
        )

    def update_types(self, **kwargs) -> bool:
        self.logger.debug(f"Node '{self.name}': Update types...")
        try:
            res = self._typefunc()
            self.logger.debug(f"Node '{self.name}': Type update is finished.")
        except Exception as exc:
            self.logger.error(
                f"Node '{self.name}': Type update failed due to exception: {exc}!"
            )
        return res

    def allocate(self, **kwargs):
        if self._allocated:
            self.logger.warning(
                f"Node '{self.name}': The memory is already allocated!"
            )
            return self._allocated
        self.logger.info(f"Node '{self.name}': Allocate the memory...")
        try:
            self._allocated = all(
                (
                    all(inp.allocate(**kwargs) for inp in self.inputs),
                    all(out.allocate(**kwargs) for out in self.outputs),
                )
            )
        except Exception as exc:
            self.logger.error(
                f"Node '{self.name}': Memory allocation failed due to "
                f"exception: {exc}"
            )
            self._allocated = False
        if self._allocated:
            self.logger.info(
                f"Node '{self.name}': Memory allocation completed successfully!"
            )
        else:
            self.logger.error(
                f"Node '{self.name}': Memory allocation failed! "
                "Inputs allocation status: "
                f"{tuple(inp._allocated for inp in self.inputs)}. "
                " Outputs allocation status: "
                f"{tuple(out._allocated for out in self.outputs)}"
            )
        return self._allocated

    def _close(self, **kwargs) -> bool:
        self.logger.debug(f"Node '{self.name}': Closing...")
        if self._closed:
            self.logger.debug(
                f"Node '{self.name}': The node is already closed!"
            )
            return self._closed
        self._closed = all(
            (
                all(inp._close(**kwargs) for inp in self.inputs),
                all(out._close(**kwargs) for out in self.outputs),
                self._allocated,
            )
        )
        if self._closed:
            self.logger.debug(
                f"Node '{self.name}': The closure completed successfully!"
            )
        else:
            self.logger.error(
                f"Node '{self.name}': The closure failed! Open inputs: "
                f"'{tuple(inp.name for inp in self.inputs if not inp.closed)}'!"
                " Open outputs: "
                f"'{tuple(out.name for out in self.outputs if not out.closed)}'!"
                f" Allocation status: {self._allocated}"
            )
        return self._closed

    def close(self, **kwargs) -> bool:
        self.logger.debug(f"Node '{self.name}': Closing...")
        if self._closed:
            self.logger.debug(
                f"Node '{self.name}': The node is already closed!"
            )
            return self._closed
        self.update_types(**kwargs)
        self._closed = all(inp.close(**kwargs) for inp in self.inputs)
        if not self._closed:
            self.logger.warning(
                f"Node '{self.name}': Some inputs are still open: "
                f"'{tuple(inp.name for inp in self.inputs if not inp.closed)}'!"
            )
            return False
        self._closed = all(out.close(**kwargs) for out in self.outputs)
        if not self._closed:
            self.logger.warning(
                f"Node '{self.name}': Some outputs are still open: "
                f"'{tuple(out.name for out in self.outputs if not out.closed)}'!"
            )
            return False
        self._closed = self.allocate(**kwargs)
        if not self._closed:
            self.logger.warning(
                f"Node '{self.name}': Some outputs are still open: "
                f"'{tuple(out.name for out in self.outputs if not out.closed)}'!"
            )
        else:
            self.logger.debug(
                f"Node '{self.name}': Closing completed successfully!"
            )
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
            return False
        self._closed = not all(out.open() for out in self.outputs)
        if self._closed:
            self.logger.warning(
                f"Node '{self.name}': Some outputs are still closed: "
                f"'{tuple(out.name for out in self.outputs if out.closed)}'!"
            )
        self.taint()
        return self._closed

