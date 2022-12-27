from .exception import (
    AllocationError,
    CriticalError,
    ClosedGraphError,
    ClosingError,
    OpeningError,
    DagflowError,
    ReconnectionError,
    UnclosedGraphError,
    InitializationError,
)
from .input import Input
from .legs import Legs
from .logger import Logger
from .output import Output, SettableOutput
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
    _being_evaluated: bool = False

    _types_tainted: bool = True

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
        elif typefunc is False:
            self._typefunc = lambda: None

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
            if (value := kwargs.pop(opt, None)) is not None:
                setattr(self, f"_{opt}", bool(value))
        if kwargs:
            raise InitializationError(f"Unparsed arguments: {kwargs}!")

    def __str__(self):
        return f"{{{self.name}}} {super().__str__()}"

    #
    # Properties
    #
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
    def types_tainted(self) -> bool:
        return self._types_tainted

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
    def being_evaluated(self) -> bool:
        return self._being_evaluated

    @property
    def allocated(self) -> bool:
        return self._allocated

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
        elif any(input.invalid for input in self.inputs):
            return
        else:
            self.invalidate_self(False)
        for output in self.outputs:
            output.invalid = invalid

    def invalidate_self(self, invalid=True) -> None:
        self._invalid = bool(invalid)
        self._frozen_tainted = False
        self._frozen = False
        self._tainted = True

    def invalidate_children(self) -> None:
        for output in self.outputs:
            output.invalid = True

    def invalidate_parents(self) -> None:
        for input in self.inputs:
            node = input.parent_node
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
            raise DagflowError("Graph is already defined")
        self._graph = graph
        self._graph.register_node(self)

    #
    # Methods
    #
    def __call__(self, name, child_output=undefined("child_output")):
        self.logger.debug(f"Node '{self.name}': Get input '{name}'...")
        inp = self.inputs.get(name, None)
        if inp is None:
            if self.closed:
                raise ClosedGraphError(node=self)
            return self._add_input(name, child_output=child_output)
        elif inp.connected and (output := inp.parent_output):
            raise ReconnectionError(input=inp, node=self, output=output)
        return inp

    def label(self, *args, **kwargs):
        if self._label:
            kwargs.setdefault("name", self._name)
            return self._label.format(*args, **kwargs)
        return self._label

    def add_input(self, name, **kwargs):
        if not self.closed:
            return self._add_input(name, **kwargs)
        raise ClosedGraphError(node=self)

    def _add_input(self, name, **kwargs):
        if IsIterable(name):
            return tuple(self._add_input(n, **kwargs) for n in name)
        self.logger.debug(f"Node '{self.name}': Add input '{name}'...")
        if name in self.inputs:
            raise ReconnectionError(input=name, node=self)
        positional = kwargs.pop("positional", True)
        keyword = kwargs.pop("keyword", True)
        inp = Input(name, self, **kwargs)
        self.inputs.add(inp, positional=positional, keyword=keyword)
        if self._graph:
            self._graph._add_input(inp)
        return inp

    def add_output(self, name, **kwargs):
        if not self.closed:
            return self._add_output(name, **kwargs)
        raise ClosedGraphError(node=self)

    def _add_output(self, name, *, settable=False, **kwargs):
        if IsIterable(name):
            return tuple(
                self._add_output(n, settable=settable, **kwargs) for n in name
            )
        self.logger.debug(f"Node '{self.name}': Add output '{name}'...")
        if isinstance(name, Output):
            if name.name in self.outputs or name.node:
                raise ReconnectionError(output=name, node=self)
            name._node = self
            return self.__add_output(
                name,
                positional=kwargs.get("positional", True),
                keyword=kwargs.get("keyword", True),
            )
        if name in self.outputs:
            raise ReconnectionError(output=name, node=self)
        output = (
            SettableOutput(name, self, **kwargs)
            if settable
            else Output(name, self, **kwargs)
        )
        return self.__add_output(
            output,
            positional=kwargs.get("positional", True),
            keyword=kwargs.get("keyword", True),
        )

    def __add_output(self, out, positional: bool = True, keyword: bool = True):
        self.outputs.add(out, positional=positional, keyword=keyword)
        if self._graph:
            self._graph._add_output(out)
        return out

    def add_pair(self, iname, oname, **kwargs):
        if not self.closed:
            return self._add_pair(iname, oname, **kwargs)
        raise ClosedGraphError(node=self)

    def _add_pair(self, iname, oname, input_kws=None, output_kws=None):
        if input_kws is None:
            input_kws = {}
        if output_kws is None:
            output_kws = {}
        output = self._add_output(oname, **output_kws)
        input = self._add_input(iname, child_output=output, **input_kws)
        return input, output

    def _wrap_fcn(self, wrap_fcn, *other_fcns):
        prev_fcn = self._stash_fcn()
        self._fcn = self._make_wrap(prev_fcn, wrap_fcn)
        if other_fcns:
            self._wrap_fcn(*other_fcns)

    def _unwrap_fcn(self):
        if not self._fcn_chain:
            raise DagflowError("Unable to unwrap bare function")
        self._fcn = self._fcn_chain.pop()

    def _stash_fcn(self):
        raise DagflowError(
            "Unimplemented method: use FunctionNode, StaticNode or MemberNode"
        )

    def _make_wrap(self, prev_fcn, wrap_fcn):
        raise DagflowError(
            "Unimplemented method: use FunctionNode, StaticNode or MemberNode"
        )

    def touch(self, force=False):
        if self._frozen:
            return
        if not self._tainted and not force:
            return
        self.logger.debug(f"Node '{self.name}': Touch...")
        ret = self.eval()
        self._tainted = False  # self._always_tainted
        if self._auto_freeze:
            self._frozen = True
        return ret

    def _eval(self):
        raise CriticalError(
            "Unimplemented method: use FunctionNode, StaticNode or MemberNode"
        )

    def eval(self):
        if not self._closed:
            raise UnclosedGraphError("Cannot evaluate the node!", node=self)
        self._being_evaluated = True
        try:
            ret = self._eval()
            self.logger.debug(f"Node '{self.name}': Evaluated return={ret}")
        except Exception as exc:
            raise exc
        self._being_evaluated = False
        return ret

    def freeze(self):
        if self._frozen:
            return
        self.logger.debug(f"Node '{self.name}': Freeze...")
        if self._tainted:
            raise CriticalError("Unable to freeze tainted node!", node=self)
        self._frozen = True
        self._frozen_tainted = False

    def unfreeze(self, force: bool = False):
        if not self._frozen and not force:
            return
        self.logger.debug(f"Node '{self.name}': Unfreeze...")
        self._frozen = False
        if self._frozen_tainted:
            self._frozen_tainted = False
            self.taint(force=True)

    def taint(self, force: bool = False):
        self.logger.debug(f"Node '{self.name}': Taint...")
        if self._tainted and not force:
            return
        if self._frozen:
            self._frozen_tainted = True
            return
        self._tainted = True
        ret = self.touch() if self._immediate else None
        self.taint_children(force)
        return ret

    def taint_children(self, force=False):
        for output in self.outputs:
            output.taint_children(force)

    def taint_type(self, force=False):
        self.logger.debug(f"Node '{self.name}': Taint types...")
        if self._closed:
            raise ClosedGraphError("Unable to taint type", node=self)
        if self._type_tainted and not force:
            return
        self._type_tainted = True
        self._tainted = True
        self._frozen = False
        for output in self.outputs:
            output.taint_children_type(force)

    def print(self):
        print(
            f"Node {self._name}: →[{len(self.inputs)}],[{len(self.outputs)}]→"
        )
        for i, input in enumerate(self.inputs):
            print("  ", i, input)
        for i, output in enumerate(self.outputs):
            print("  ", i, output)

    def _typefunc(self) -> bool:
        """A output takes this function to determine the dtype and shape"""
        raise DagflowError(
            "Unimplemented method: the method must be overridden!"
        )

    def update_types(self, recursive: bool = True) -> bool:
        if not self._types_tainted:
            return True
        self.logger.debug(f"Node '{self.name}': Update types...")
        if recursive:
            for input in self.inputs:
                input.parent_node.update_types(recursive)
        self._typefunc()
        self._types_tainted = False

    def allocate(self, recursive: bool = True):
        if self._allocated:
            return True
        self.logger.debug(f"Node '{self.name}': Allocate memory...")
        if recursive and not all(
            input.parent_node.allocate(recursive) for input in self.inputs
        ):
            return False
        if not self.inputs.allocate():
            raise AllocationError(
                "Cannot allocate memory for inputs!", node=self
            )
        if not self.outputs.allocate():
            raise AllocationError(
                "Cannot allocate memory for outputs!", node=self
            )
        self.post_allocate()
        self._allocated = True
        return True

    def post_allocate(self):
        pass

    def close(self, recursive: bool = True) -> bool:
        if self._closed:
            return True
        self.logger.debug(f"Node '{self.name}': Close...")
        if self.invalid:
            raise ClosingError("Cannot close an invalid node!", node=self)
        if recursive and not all(
            input.parent_node.close(recursive) for input in self.inputs
        ):
            return False
        self.update_types(recursive=recursive)
        self.allocate(recursive=recursive)
        self._closed = self._allocated
        if not self._closed:
            raise ClosingError(node=self)
        return self._closed

    def open(self, force: bool = False) -> bool:
        if not self._closed and not force:
            return True
        self.logger.debug(f"Node '{self.name}': Open...")
        if not all(
            input.node.open(force)
            for output in self.outputs
            for input in output.child_inputs
        ):
            raise OpeningError(node=self)
        self.unfreeze()
        self.taint()
        self._closed = False
        return not self._closed

    #
    # Accessors
    #
    def get_data(self, key):
        return self.outputs[key].data

    def get_input_data(self, key):
        return self.inputs[key].data()
