from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from weakref import ReferenceType
from weakref import ref as weakref

from .exception import (
    AllocationError,
    ClosedGraphError,
    ClosingError,
    CriticalError,
    DagflowError,
    InitializationError,
    OpeningError,
    ReconnectionError,
    UnclosedGraphError,
)
from .input import Input
from .iter import IsIterable
from .labels import Labels
from .limbs import Limbs
from .logger import Logger, get_logger
from .output import Output
from .types import GraphT

if TYPE_CHECKING:
    from .meta_node import MetaNode
    from .storage import NodeStorage


class Node(Limbs):
    __slots__ = (
        "_name",
        "_labels",
        "_graph",
        "_logger",
        "_exception",
        "_meta_node",
        "_tainted",
        "_frozen",
        "_frozen_tainted",
        "_invalid",
        "_types_tainted",
        "_auto_freeze",
        "_immediate",
        "_closed",
        "_allocated",
        "_being_evaluated",
        "_debug",
    )

    _name: str
    _labels: Labels
    _graph: Optional[GraphT]
    _exception: Optional[str]

    _meta_node: Optional[ReferenceType]

    # Taintflag and status
    _tainted: bool
    _frozen: bool
    _frozen_tainted: bool
    _invalid: bool
    _closed: bool
    _allocated: bool
    _being_evaluated: bool

    _types_tainted: bool

    # Options
    _debug: bool
    _auto_freeze: bool
    _immediate: bool
    # _always_tainted: bool

    def __init__(
        self,
        name,
        *,
        label: Union[str, dict, None] = None,
        graph: Optional[GraphT] = None,
        debug: Optional[bool] = None,
        logger: Optional[Any] = None,
        missing_input_handler: Optional[Callable] = None,
        immediate: bool = False,
        auto_freeze: bool = False,
        frozen: bool = False,
        **kwargs,
    ):
        super().__init__(missing_input_handler=missing_input_handler)
        self._graph = None
        self._logger = None
        self._exception = None
        self._meta_node = None

        self._tainted = True
        self._frozen = False
        self._frozen_tainted = False
        self._invalid = False
        self._closed = False
        self._allocated = False
        self._being_evaluated = False
        self._types_tainted = True
        self._auto_freeze = False
        self._immediate = False

        self._name = name

        if graph is None:
            from .graph import Graph

            self.graph = Graph.current()
        else:
            self.graph = graph

        if debug is None and self.graph is not None:
            self._debug = self.graph.debug
        else:
            self._debug = bool(debug)

        self._labels = Labels(label or name)

        if logger is not None:
            self._logger = logger
        elif self.graph is not None:
            self._logger = self.graph.logger
        else:
            self._logger = get_logger()

        self._immediate = immediate
        self._auto_freeze = auto_freeze
        self._frozen = frozen

        if kwargs:
            raise InitializationError(f"Unparsed arguments: {kwargs}!")

    @classmethod
    def make_stored(
        cls, name: str, *args, label_from: Optional[Mapping] = None, **kwargs
    ) -> Tuple[Optional["Node"], "NodeStorage"]:
        from multikeydict.nestedmkdict import NestedMKDict

        if label_from is not None:
            label_from = NestedMKDict(label_from, sep=".")
            try:
                label = label_from.any(name, object=True)
            except KeyError:
                raise RuntimeError(f"Could not find label for {name}")
            kwargs.setdefault("label", label)

        node = cls(name, *args, **kwargs)

        from .storage import NodeStorage
        storage = NodeStorage(default_containers=True)
        storage("nodes")[name] = node
        if len(node.outputs) == 1:
            storage("outputs")[name] = node.outputs[0]

        NodeStorage.update_current(storage, strict=True)

        return node, storage

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
    def exception(self):
        return self._exception

    @property
    def meta_node(self) -> "MetaNode":
        return self._meta_node and self._meta_node()

    @meta_node.setter
    def meta_node(self, value: "MetaNode"):
        if self._meta_node is not None:
            raise RuntimeError("MetaNode may be set only once")
        self._meta_node = weakref(value)

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
        elif any(_input.invalid for _input in self.inputs.iter_all()):
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
        for _input in self.inputs.iter_all():
            node = _input.parent_node
            node.invalidate_self()
            node.invalidate_parents()

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, graph):
        if graph is None:
            return
        if self._graph is not None:
            raise DagflowError("Graph is already defined")
        self._graph = graph
        self._graph.register_node(self)

    @property
    def labels(self) -> Labels:
        return self._labels

    #
    # Methods
    #
    def __call__(self, name: Optional[str] = None, *args, **kwargs) -> Optional[Input]:
        """
        Returns an existing input by `name`, else try to create new one.
        If `name` is given, creates an input by the default way,
        otherwise tries to use `missing_input_handler`.

        .. note:: creation of a new input is restricted for a *closed* graph
        """
        if name is None:
            self.logger.debug(
                f"Node '{self.name}': Try to create an input with `missing_input_handler`"
            )
            if not self.closed:
                return self._make_input(*args, **kwargs)
            raise ClosedGraphError(node=self)

        self.logger.debug(
            f"Node '{self.name}': Try to get or create the input '{name}'"
        )
        kwargs.setdefault("positional", False)
        inp = self.inputs.get(name, None)
        if inp is None:
            if self.closed:
                raise ClosedGraphError(node=self)
            return self._add_input(name, **kwargs)
        elif inp.connected and (output := inp.parent_output):
            raise ReconnectionError(input=inp, node=self, output=output)
        return inp

    def _make_input(self, *args, exception=True, **kwargs) -> Optional[Input]:
        handler = self._missing_input_handler
        if handler is None:
            if exception:
                raise RuntimeError(
                    "Unable to make an input automatically as no handler is set"
                )
            return None
        return handler(*args, **kwargs)

    def label(self) -> Optional[str]:
        return self._labels.text

    def add_input(
        self, name: Union[str, Sequence[str]], **kwargs
    ) -> Union[Input, Tuple[Input]]:
        if self.closed:
            raise ClosedGraphError(node=self)
        if isinstance(name, str):
            return self._add_input(name, **kwargs)
        if IsIterable(name):
            return tuple(self._add_input(n, **kwargs) for n in name)
        raise CriticalError(
            f"'name' of the input must be `str` or `Sequence[str]`, but given {name}",
            node=self,
        )

    def _add_input(
        self,
        name: str,
        *,
        positional: bool = True,
        keyword: bool = True,
        **kwargs,
    ) -> Input:
        self.logger.debug(f"Node '{self.name}': Add input '{name}'")
        if name in self.inputs:
            raise ReconnectionError(input=name, node=self)
        inp = Input(name, self, **kwargs)
        self.inputs.add(inp, positional=positional, keyword=keyword)
        if self._graph:
            self._graph._add_input(inp)
        return inp

    def add_output(
        self, name: Union[str, Sequence[str]], **kwargs
    ) -> Union[Output, Tuple[Output]]:
        if self.closed:
            raise ClosedGraphError(node=self)
        if isinstance(name, str):
            return self._add_output(name, **kwargs)
        if IsIterable(name):
            return tuple(self._add_output(n, **kwargs) for n in name)
        raise CriticalError(
            f"'name' of the output must be `str` or `Sequence[str]`, but given {name=}",
            node=self,
        )

    def _add_output(
        self,
        name: str,
        *,
        keyword: bool = True,
        positional: bool = True,
        **kwargs,
    ) -> Output:
        self.logger.debug(f"Node '{self.name}': Add output '{name}'")
        if name in self.outputs:
            raise ReconnectionError(output=name, node=self)
        out = Output(name, self, **kwargs)
        self.outputs.add(out, positional=positional, keyword=keyword)
        if self._graph:
            self._graph._add_output(out)
        return out

    def add_pair(
        self, iname: str, oname: str, **kwargs
    ) -> Tuple[Union[Input, Tuple[Input]], Union[Output, Tuple[Output]]]:
        if self.closed:
            raise ClosedGraphError(node=self)
        return self._add_pair(iname, oname, **kwargs)

    def _add_pair(
        self,
        iname: str,
        oname: str,
        input_kws: Optional[dict] = None,
        output_kws: Optional[dict] = None,
    ) -> Tuple[Union[Input, Tuple[Input]], Union[Output, Tuple[Output]]]:
        input_kws = input_kws or {}
        output_kws = output_kws or {}
        output = self.add_output(oname, **output_kws)
        child_output = output if isinstance(output, Output) else None
        input = self.add_input(iname, child_output=child_output, **input_kws)
        return input, output

    def touch(self, force=False):
        if self._frozen:
            return
        if not self._tainted and not force:
            return
        self.logger.debug(f"Node '{self.name}': Touch")
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
        self.logger.debug(f"Node '{self.name}': Freeze")
        if self._tainted:
            raise CriticalError("Unable to freeze tainted node!", node=self)
        self._frozen = True
        self._frozen_tainted = False

    def unfreeze(self, force: bool = False):
        if not self._frozen and not force:
            return
        self.logger.debug(f"Node '{self.name}': Unfreeze")
        self._frozen = False
        if self._frozen_tainted:
            self._frozen_tainted = False
            self.taint(force=True)

    def taint(self, *, caller: Optional[Input] = None, force: bool = False):
        self.logger.debug(f"Node '{self.name}': Taint...")
        if self._tainted and not force:
            return
        if self._frozen:
            self._frozen_tainted = True
            return
        self._tainted = True
        self._on_taint(caller)
        ret = self.touch() if self._immediate else None
        self.taint_children(force=force)
        return ret

    def taint_children(self, **kwargs):
        for output in self.outputs:
            output.taint_children(**kwargs)

    def taint_type(self, force: bool = False):
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
        print(f"Node {self._name}: →[{len(self.inputs)}],[{len(self.outputs)}]→")
        for i, _input in enumerate(self.inputs):
            print("  ", i, _input)
        for _input in self.inputs.iter_nonpos():
            print("    ", _input)
        for i, output in enumerate(self.outputs):
            print("  ", i, output)
        for output in self.outputs.iter_nonpos():
            print("    ", output)

    def _typefunc(self) -> bool:
        """A output takes this function to determine the dtype and shape"""
        raise DagflowError("Unimplemented method: the method must be overridden!")

    def _on_taint(self, caller: Input):
        """A node method to be called on taint"""

    def _post_allocate(self):
        pass

    def update_types(self, recursive: bool = True):
        if not self._types_tainted:
            return True
        if recursive:
            self.logger.debug(f"Node '{self.name}': Trigger recursive update types...")
            for input in self.inputs.iter_all():
                if not input.connected():
                    raise ClosingError('Input is not connected', node=self, input=input)
                input.parent_node.update_types(recursive)
        self.logger.debug(f"Node '{self.name}': Update types...")
        self._typefunc()
        self._types_tainted = False

    def allocate(self, recursive: bool = True):
        if self._allocated:
            return True
        if recursive:
            self.logger.debug(
                f"Node '{self.name}': Trigger recursive memory allocation..."
            )
            if not all(
                _input.parent_node.allocate(recursive)
                for _input in self.inputs.iter_all()
            ):
                return False
        self.logger.debug(f"Node '{self.name}': Allocate memory on inputs")
        if not self.inputs.allocate():
            raise AllocationError("Cannot allocate memory for inputs!", node=self)
        self.logger.debug(f"Node '{self.name}': Allocate memory on outputs")
        if not self.outputs.allocate():
            raise AllocationError("Cannot allocate memory for outputs!", node=self)
        self.logger.debug(f"Node '{self.name}': Post allocate")
        self._post_allocate()
        self._allocated = True
        return True

    def close(self, recursive: bool = True, together: Sequence["Node"] = []) -> bool:
        # Caution: `together` list should not be written in!

        if self._closed:
            return True
        if self.invalid:
            raise ClosingError("Cannot close an invalid node!", node=self)
        self.logger.debug(f"Node '{self.name}': Trigger recursive close")
        for node in [self] + together:
            node.update_types(recursive=recursive)
        for node in [self] + together:
            node.allocate(recursive=recursive)
        if recursive and not all(
            _input.parent_node.close(recursive) for _input in self.inputs.iter_all()
        ):
            return False
        for node in together:
            if not node.close(recursive=recursive):
                return False
        self._closed = self._allocated
        if not self._closed:
            raise ClosingError(node=self)
        self.logger.debug(f"Node '{self.name}': Closed")
        return self._closed

    def open(self, force: bool = False) -> bool:
        if not self._closed and not force:
            return True
        self.logger.debug(f"Node '{self.name}': Open")
        if not all(
            _input.node.open(force)
            for output in self.outputs
            for _input in output.child_inputs
        ):
            raise OpeningError(node=self)
        self.unfreeze()
        self.taint()
        self._closed = False
        return not self._closed
