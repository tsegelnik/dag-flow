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

from multikeydict.typing import KeyLike
from .flagsdescriptor import FlagsDescriptor

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
from .nodebase import NodeBase
from .logger import Logger, get_logger
from .output import Output
from .types import GraphT

if TYPE_CHECKING:
    from .metanode import MetaNode
    from .storage import NodeStorage


class Node(NodeBase):
    __slots__ = (
        "_name",
        "_labels",
        "_graph",
        "_logger",
        "_exception",
        "_metanode",
        "_auto_freeze",
        "_immediate",
        "_debug",
        "_allowed_kw_inputs",
        "_fd",
    )

    _name: str
    _labels: Labels
    _allowed_kw_inputs: Tuple[str, ...]
    _graph: Optional[GraphT]
    _exception: Optional[str]

    _metanode: Optional[ReferenceType]
    _fd: FlagsDescriptor

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
        allowed_kw_inputs: Sequence[str] = (),
        **kwargs,
    ):
        super().__init__(missing_input_handler=missing_input_handler)
        self._graph = None
        self._logger = None
        self._exception = None
        self._metanode = None

        self._name = name
        self._allowed_kw_inputs = tuple(allowed_kw_inputs)
        self._name = name
        self._fd = FlagsDescriptor(**kwargs)

        if graph is None:
            from .graph import Graph  # fmt:skip
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
        self._fd.frozen = frozen

        if kwargs:
            raise InitializationError(f"Unparsed arguments: {kwargs}!")

    def __str__(self):
        return f"{{{self.name}}} {super().__str__()}"

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

    @classmethod
    def replicate(
        cls,
        name: str,
        replicate: Tuple[KeyLike, ...] = ((),),
        **kwargs,
    ) -> Tuple[Optional["Node"], "NodeStorage"]:
        from .storage import NodeStorage

        storage = NodeStorage(default_containers=True)
        nodes = storage("nodes")
        inputs = storage("inputs")
        outputs = storage("outputs")

        if not replicate:
            raise RuntimeError("`replicate` tuple should have at least one item")

        tuplename = (name,)
        for key in replicate:
            if isinstance(key, str):
                key = (key,)
            outname = tuplename + key
            instance = cls(".".join(outname), **kwargs)
            nodes[outname] = instance

            ninputs = instance.inputs.len_all()
            noutputs = instance.outputs.len_all()
            if noutputs == 0:
                instance()
                ninputs = instance.inputs.len_all()
                noutputs = instance.outputs.len_all()

            iter_inputs = instance.inputs.iter_all_items()
            if ninputs > 1:
                for iname, input in iter_inputs:
                    inputs[tuplename + (iname,) + key] = input
            else:
                _, input = next(iter_inputs)
                inputs[tuplename + key] = input

            iter_outputs = instance.outputs.iter_all_items()
            if noutputs > 1:
                for oname, output in instance.outputs.iter_all_items():
                    outputs[tuplename + (oname,) + key] = output
            else:
                _, output = next(iter_outputs)
                outputs[tuplename + key] = output

        NodeStorage.update_current(storage, strict=True)

        if len(replicate) == 1:
            return instance, storage

        return None, storage

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
    def allowed_kw_inputs(self) -> Tuple[str]:
        return self._allowed_kw_inputs

    @property
    def exception(self):
        return self._exception

    @property
    def metanode(self) -> "MetaNode":
        return self._metanode and self._metanode()

    @metanode.setter
    def metanode(self, value: "MetaNode"):
        if self._metanode is not None:
            raise RuntimeError("MetaNode may be set only once")
        self._metanode = weakref(value)

    @property
    def logger(self) -> Logger:
        return self._logger

    @property
    def tainted(self) -> bool:
        return self._fd.tainted

    @property
    def types_tainted(self) -> bool:
        return self._fd.types_tainted

    @property
    def frozen_tainted(self) -> bool:
        return self._fd.frozen_tainted

    @property
    def frozen(self) -> bool:
        return self._fd.frozen

    @property
    def auto_freeze(self) -> bool:
        return self._auto_freeze

    # @property
    # def always_tainted(self) -> bool:
    # return self._fd.always_tainted

    @property
    def closed(self) -> bool:
        return self._fd.closed

    @property
    def debug(self) -> bool:
        return self._debug

    @property
    def being_evaluated(self) -> bool:
        return self._fd.being_evaluated

    @property
    def allocated(self) -> bool:
        return self._fd.allocated

    @property
    def immediate(self) -> bool:
        return self._immediate

    @property
    def invalid(self) -> bool:
        return self._fd.invalid

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
        self._fd.invalid = bool(invalid)
        self._fd.frozen_tainted = False
        self._fd.frozen = False
        self._fd.tainted = True

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

    def label(self) -> Optional[str]:
        return self._labels.text

    #
    # Methods
    #
    def __call__(self, name: Optional[str] = None, *args, **kwargs) -> Optional[Input]:
        """
        Returns an existing input by `name`, else try to create new one.
        If `name` is given, creates an input by the default way,
        otherwise tries to use `missing_input_handler`.
        If `name` is not given simply uses the input handler.

        .. note:: creation of a new input is restricted for a *closed* graph
        """
        if name is None:
            self.logger.debug(
                f"Node '{self.name}': Try to create an input with `missing_input_handler`"
            )
            if not self.closed:
                return self._make_input(*args, **kwargs)
            raise ClosedGraphError(node=self)

        self.logger.debug(f"Node '{self.name}': Try to get/create the input '{name}'")
        inp = self.inputs.get(name, None)
        kwargs.setdefault("positional", False)
        if inp is None:
            inp = self.add_input(name, **kwargs)
        elif isinstance(inp, Input) and inp.connected and (output := inp.parent_output):
            raise ReconnectionError(input=inp, node=self, output=output)
        return inp

    def _make_input(self, *args, exception=True, **kwargs) -> Optional[Input]:
        """
        Creates a single input via an input handler
        """
        handler = self._missing_input_handler
        if handler is None:
            if exception:
                raise RuntimeError("Unable to make an input automatically as no handler is set")
            return None
        return handler(*args, **kwargs)

    def add_input(self, name: str, **kwargs) -> Input:
        """
        Creates a single input with name, if the graph is not closed.

        If the node has `allowed_kw_inputs` checks whether the name in the `allowed_kw_inputs`
        and then directly creates a new input.
        Else the node has no `allowed_kw_inputs` firstly tries to use an input handler.
        If there is no input handler the method creates a new input directly.
        """
        if self.closed:
            raise ClosedGraphError(node=self)
        if self.allowed_kw_inputs:
            if name not in self.allowed_kw_inputs:
                raise CriticalError(
                    (
                        f"Cannot create an input with {name=} due to the name is not in the "
                        f"allowed_kw_inputs={self.allowed_kw_inputs}"
                    ),
                    node=self,
                )
            return self._add_input(name, **kwargs)
        if (inp := self._make_input(exception=False)) is None:
            inp = self._add_input(name, **kwargs)
        return inp

    def _add_inputs(self, name: Sequence[str], **kwargs) -> Tuple[Input, ...]:
        """
        Creates a sequence of inputs

        .. note:: there is no check whether the graph is closed or not.
        """
        if IsIterable(name):
            return tuple(self._add_input(n, **kwargs) for n in name)
        raise CriticalError(
            f"'name' of the input must be `Sequence[str]`, but given {name}",
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
        """
        Creates a new single input if there is no input with `name`.

        .. note:: there is no check whether the graph is closed or not.
        """
        self.logger.debug(f"Node '{self.name}': Add input '{name}'")
        if name in self.inputs:
            raise ReconnectionError(input=name, node=self)
        inp = Input(name, self, **kwargs)
        self.inputs.add(inp, positional=positional, keyword=keyword)
        if self._graph:
            self._graph._add_input(inp)
        return inp

    def add_output(
        self,
        name: str,
        *,
        keyword: bool = True,
        positional: bool = True,
        **kwargs,
    ) -> Output:
        """Creates a new single output if there is no output with `name` and the graph is not closed"""
        if self.closed:
            raise ClosedGraphError(node=self)
        return self._add_output(name, keyword=keyword, positional=positional, **kwargs)

    def _add_outputs(self, name: Sequence[str], **kwargs) -> Tuple[Output, ...]:
        """
        Creates a sequence of outputs

        .. note:: there is no check whether the graph is closed or not.
        """
        if IsIterable(name):
            return tuple(self._add_output(n, **kwargs) for n in name)
        raise CriticalError(
            f"'name' of the output must be `Sequence[str]`, but given {name=}",
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
        """
        Creates a new single output if there is no output with `name`.

        .. note:: there is no check whether the graph is closed or not.
        """
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
        """
        Creates a pair of input and output
        """
        if self.closed:
            raise ClosedGraphError(node=self)
        return self._add_pair(iname, oname, **kwargs)

    def _add_pairs(
        self,
        inames: Sequence[str],
        onames: Sequence[str],
        input_kws: Optional[dict] = None,
        output_kws: Optional[dict] = None,
    ) -> Tuple[List[Input], List[Output]]:
        """
        Creates sequence of pairs of input and output

        .. note:: the inputs and outputs count must be the same
        """
        if len(inames) != len(onames):
            raise CriticalError(
                (
                    f"Cannot add pairs of input/output due to different lenght of {inames=} and"
                    f" {onames=}"
                ),
                node=self,
            )
        inputs, outputs = [], []
        for iname, oname in zip(inames, onames):
            input, output = self._add_pair(iname, oname, input_kws, output_kws)
            inputs.append(input)
            outputs.append(output)
        return inputs, outputs

    def _add_pair(
        self,
        iname: str,
        oname: str,
        input_kws: Optional[dict] = None,
        output_kws: Optional[dict] = None,
    ) -> Tuple[Union[Input, Tuple[Input]], Union[Output, Tuple[Output]]]:
        """
        Creates a pair of input and output

        .. note:: there is no check whether the graph is closed or not.
        """
        input_kws = input_kws or {}
        output_kws = output_kws or {}
        output = self._add_output(oname, **output_kws)
        input = self._add_input(iname, child_output=output, **input_kws)
        return input, output

    def touch(self, force=False):
        if self.frozen:
            return
        if not self.tainted and not force:
            return
        self.logger.debug(f"Node '{self.name}': Touch")
        ret = self.eval()
        self._fd.tainted = False  # self._always_tainted
        if self._auto_freeze:
            self._fd.frozen = True
        return ret

    def _eval(self):
        raise CriticalError("Unimplemented method: use FunctionNode, StaticNode or MemberNode")

    def eval(self):
        if not self.closed:
            raise UnclosedGraphError("Cannot evaluate the node!", node=self)
        self._fd.being_evaluated = True
        try:
            ret = self._eval()
            self.logger.debug(f"Node '{self.name}': Evaluated return={ret}")
        except Exception as exc:
            raise exc
        return ret

    def freeze(self):
        if self._fd.frozen:
            return
        self.logger.debug(f"Node '{self.name}': Freeze")
        if self.tainted:
            raise CriticalError("Unable to freeze tainted node!", node=self)
        self._fd.frozen = True
        self._fd.frozen_tainted = False

    def unfreeze(self, force: bool = False):
        if not self._fd.frozen and not force:
            return
        self.logger.debug(f"Node '{self.name}': Unfreeze")
        self._fd.frozen = False
        if self.frozen_tainted:
            self._fd.frozen_tainted = False
            self.taint(force=True)

    def taint(self, *, caller: Optional[Input] = None, force: bool = False):
        self.logger.debug(f"Node '{self.name}': Taint...")
        if self._fd.tainted and not force:
            return
        if self.frozen:
            self._fd.frozen_tainted = True
            return
        self._fd.tainted = True
        self._on_taint(caller)
        ret = self.touch() if (self._immediate or force) else None
        self.taint_children(force=force)
        return ret

    def taint_children(self, **kwargs):
        for output in self.outputs:
            output.taint_children(**kwargs)

    def taint_type(self, force: bool = False):
        self.logger.debug(f"Node '{self.name}': Taint types...")
        if self._fd.closed:
            raise ClosedGraphError("Unable to taint type", node=self)
        if self.types_tainted and not force:
            return
        self._fd.types_tainted = True
        self._fd.tainted = True
        self._fd.frozen = False
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
        if not self.types_tainted:
            return True
        if recursive:
            self.logger.debug(f"Node '{self.name}': Trigger recursive update types...")
            for input in self.inputs.iter_all():
                if not input.connected():
                    raise ClosingError("Input is not connected", node=self, input=input)
                input.parent_node.update_types(recursive)
        self.logger.debug(f"Node '{self.name}': Update types...")
        self._typefunc()
        self._fd.types_tainted = False

    def allocate(self, recursive: bool = True):
        if self._fd.allocated:
            return True
        if recursive:
            self.logger.debug(f"Node '{self.name}': Trigger recursive memory allocation...")
            for _input in self.inputs.iter_all():
                try:
                    parent_node = _input.parent_node
                except AttributeError:
                    raise ClosingError("Parent node is not initialized", input=_input)
                if not parent_node.allocate(recursive):
                    return False
        self.logger.debug(f"Node '{self.name}': Allocate memory on inputs")
        if not self.inputs.allocate():
            raise AllocationError("Cannot allocate memory for inputs!", node=self)
        self.logger.debug(f"Node '{self.name}': Allocate memory on outputs")
        if not self.outputs.allocate():
            raise AllocationError("Cannot allocate memory for outputs!", node=self)
        self.logger.debug(f"Node '{self.name}': Post allocate")
        self._post_allocate()
        self._fd.allocated = True
        return True

    def close(
        self,
        recursive: bool = True,
        strict: bool = True,
        together: Sequence["Node"] = [],
    ) -> bool:
        # Caution: `together` list should not be written in!

        if self._fd.closed:
            return True
        if self.invalid:
            raise ClosingError("Cannot close an invalid node!", node=self)
        self.logger.debug(f"Node '{self.name}': Trigger recursive close")
        for node in [self] + together:
            try:
                node.update_types(recursive=recursive)
            except ClosingError:
                if strict:
                    raise
        for node in [self] + together:
            try:
                node.allocate(recursive=recursive)
            except ClosingError:
                if strict:
                    raise
        if recursive and not all(
            _input.parent_node.close(recursive) for _input in self.inputs.iter_all()
        ):
            return False
        for node in together:
            if not node.close(recursive=recursive):
                return False
        self._fd.closed = self._fd.allocated
        if strict and not self.closed:
            raise ClosingError(node=self)
        self.logger.debug(f"Node '{self.name}': {self.closed and 'closed' or 'failed to close'}")
        return self.closed

    def open(self, force: bool = False) -> bool:
        if not self.closed and not force:
            return True
        self.logger.debug(f"Node '{self.name}': Open")
        if not all(
            _input.node.open(force) for output in self.outputs for _input in output.child_inputs
        ):
            raise OpeningError(node=self)
        self.unfreeze()
        self.taint()
        self._fd.closed = False
        return not self.closed
