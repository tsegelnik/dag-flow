from __future__ import annotations

from typing import TYPE_CHECKING, Callable
from weakref import ref as weakref

from multikeydict.typing import KeyLike, properkey

from ..core.labels import Labels
from ..tools.logger import Logger, get_logger
from .exception import (
    ClosedGraphError,
    ClosingError,
    CriticalError,
    DagflowError,
    InitializationError,
    OpeningError,
    ReconnectionError,
    UnclosedGraphError,
)
from .flags_descriptor import FlagsDescriptor
from .graph import Graph
from .input import Input
from .iter import IsIterable
from .node_base import NodeBase
from .output import Output

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any
    from weakref import ReferenceType

    from .meta_node import MetaNode
    from .storage import NodeStorage


class Node(NodeBase):
    __slots__ = (
        "_name",
        "_labels",
        "_graph",
        "_logger",
        "_exception",
        "_meta_node",
        "_immediate",
        "_debug",
        "_allowed_kw_inputs",
        "_fd",
        "function",
        "_functions_dict",
        "_n_calls",
        "_input_nodes_callbacks",
    )

    _name: str
    _labels: Labels
    _allowed_kw_inputs: tuple[str, ...]
    _graph: Graph | None
    _exception: str | None
    _logger: Logger

    _meta_node: ReferenceType | None
    _fd: FlagsDescriptor
    _n_calls: int

    # Options
    _debug: bool
    _immediate: bool

    _input_nodes_callbacks: list[Callable]

    def __init__(
        self,
        name,
        *,
        label: str | dict | None = None,
        graph: Graph | None = None,
        debug: bool | None = None,
        logger: Any | None = None,
        missing_input_handler: Callable | None = None,
        immediate: bool = False,
        frozen: bool = False,
        allowed_kw_inputs: Sequence[str] = (),
        **kwargs,
    ):
        super().__init__(missing_input_handler=missing_input_handler)
        self._graph = None
        self._exception = None
        self._meta_node = None

        self._name = name
        self._allowed_kw_inputs = tuple(allowed_kw_inputs)
        self._fd = FlagsDescriptor(children=self.outputs, parents=self.inputs, **kwargs)
        self._n_calls = 0

        self.graph = Graph.current() if graph is None else graph
        if debug is None and self.graph is not None:
            self._debug = self.graph.debug
        else:
            self._debug = bool(debug)

        self._labels = Labels(label or name)

        if isinstance(logger, Logger):
            self._logger = logger
        elif logger is not None:
            raise InitializationError(f"Cannot initialize a node with logger={logger}", node=self)
        elif self.graph is not None:
            self._logger = self.graph.logger
        else:
            self._logger = get_logger()

        self._immediate = immediate
        self.fd.frozen = frozen

        self._functions_dict: dict[Any, Callable] = {"default": self._function}
        self.function = self._functions_dict["default"]

        self._input_nodes_callbacks = []

        if kwargs:
            raise InitializationError(f"Unparsed arguments: {kwargs}!")

    def __str__(self):
        return f"{{{self.name}}} {super().__str__()}"

    @classmethod
    def from_args(
        cls, name, *positional_connectibles, kwargs: Mapping = {}, **key_connectibles
    ) -> Node:
        # TODO:
        #   - testing
        #   - keyword connection syntax ([] or ())
        instance = cls(name, **kwargs)
        for connectible in positional_connectibles:
            connectible >> instance
        for key, connectible in key_connectibles.items():
            connectible >> instance.inputs[key]
        return instance

    @classmethod
    def replicate(
        cls,
        *,
        name: str,
        replicate_outputs: tuple[KeyLike, ...] = ((),),
        **kwargs,
    ) -> tuple[Node | None, "NodeStorage"]:
        from .storage import NodeStorage

        storage = NodeStorage(default_containers=True)
        nodes = storage("nodes")
        inputs = storage("inputs")
        outputs = storage("outputs")

        if not replicate_outputs:
            raise RuntimeError("`replicate_outputs` tuple should have at least one item")

        tuplename = (name,)
        for key in replicate_outputs:
            key = properkey(key)
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
            elif ninputs == 1:
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

        return (instance, storage) if len(replicate_outputs) == 1 else (None, storage)

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
    def allowed_kw_inputs(self) -> tuple[str, ...]:
        return self._allowed_kw_inputs

    @property
    def exception(self):
        return self._exception

    @property
    def meta_node(self) -> MetaNode:
        return self._meta_node and self._meta_node()

    @meta_node.setter
    def meta_node(self, value: MetaNode):
        if self._meta_node is not None:
            raise RuntimeError("MetaNode may be set only once")
        self._meta_node = weakref(value)

    @property
    def logger(self) -> Logger:
        return self._logger

    @property
    def tainted(self) -> bool:
        return self.fd.tainted

    @property
    def frozen_tainted(self) -> bool:
        return self.fd.frozen_tainted

    @property
    def frozen(self) -> bool:
        return self.fd.frozen

    @property
    def closed(self) -> bool:
        return self.fd.closed

    @property
    def debug(self) -> bool:
        return self._debug

    @property
    def being_evaluated(self) -> bool:
        return self.fd.being_evaluated

    @property
    def allocated(self) -> bool:
        return self.fd.allocated

    @property
    def immediate(self) -> bool:
        return self._immediate

    @property
    def invalid(self) -> bool:
        return self.fd.invalid

    @property
    def fd(self) -> FlagsDescriptor:
        return self._fd

    @invalid.setter
    def invalid(self, invalid: bool) -> None:
        self.fd._invalidate(invalid)

    def invalidate(self, invalid: bool = True) -> None:
        return self.fd.invalidate(invalid)

    def invalidate_children(self, invalid: bool = True) -> None:
        return self.fd.invalidate_children(invalid)

    def invalidate_parents(self, invalid: bool = True) -> None:
        return self.fd.invalidate_parents(invalid)

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

    def label(self) -> str | None:
        return self._labels.text

    @property
    def n_calls(self) -> int:
        return self._n_calls

    #
    # Methods
    #
    def __call__(self, name: str | None = None, *args, **kwargs) -> Input | None:
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
        inp: Any = self.inputs.get(name, None)
        kwargs.setdefault("positional", False)
        if inp is None:
            inp = self.add_input(name, **kwargs)
        elif isinstance(inp, Input) and inp.connected and (output := inp.parent_output):
            raise ReconnectionError(input=inp, node=self, output=output)
        return inp

    def _make_input(self, *args, exception=True, **kwargs) -> Input | None:
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

    def _add_inputs(self, name: Sequence[str], **kwargs) -> tuple[Input, ...]:
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
        self._fd.allocated = False
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
        self._fd.allocated = False
        return out

    def _add_outputs(self, name: Sequence[str], **kwargs) -> tuple[Output, ...]:
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

    def add_pair(
        self, iname: str, oname: str, **kwargs
    ) -> tuple[Input | tuple[Input, ...], Output | tuple[Output, ...]]:
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
        input_kws: dict | None = None,
        output_kws: dict | None = None,
    ) -> tuple[list[Input], list[Output]]:
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
        input_kws: dict | None = None,
        output_kws: dict | None = None,
    ) -> tuple[Input | tuple[Input, ...], Output | tuple[Output, ...]]:
        """
        Creates a pair of input and output

        .. note:: there is no check whether the graph is closed or not.
        """
        input_kws = input_kws or {}
        output_kws = output_kws or {}
        out = self._add_output(oname, **output_kws)
        inp = self._add_input(iname, child_output=out, **input_kws)
        self._fd.allocated = False
        return inp, out

    def _function(self):
        pass

    def eval(self):
        if not self.closed:
            raise UnclosedGraphError("Cannot evaluate not closed node!", node=self)
        self._eval()

    def _eval(self):
        self.fd.being_evaluated = True
        self._n_calls += 1
        self.function()
        self.fd.being_evaluated = False

    def touch(self, force_computation=False):
        if not force_computation:
            if not self.tainted:
                return
            if not self.closed:
                raise UnclosedGraphError("Cannot evaluate not closed node!", node=self)
        self._touch()

    def _touch(self):
        # To avoid extra function calls we copy lines below from _eval
        self.fd.being_evaluated = True
        self._n_calls += 1
        self.function()
        self.fd.being_evaluated = False
        self.fd.tainted = False

    def freeze(self):
        if self.frozen:
            return
        if self.tainted:
            raise CriticalError("Unable to freeze tainted node!", node=self)
        self.fd.freeze()

    def unfreeze(self):
        if not self.frozen:
            return
        self.fd.frozen = False
        if self.frozen_tainted:
            self.fd.frozen_tainted = False
            self.taint()

    def taint(
        self, *, caller: Input | None = None, force: bool = False, force_computation: bool = False
    ):
        self._on_taint(caller)
        if self.tainted and not force:
            return
        if self.frozen:
            self.fd.frozen_tainted = True
            return

        self.fd.tainted = True
        ret = self._touch() if (self._immediate or force_computation) else None
        # TODO:  maybe here it is better to avoid extra call from FlagsDescriptor
        self.fd.taint_children(force=force)

        return ret

    def taint_children(self, **kwargs):
        self.fd.taint_children(**kwargs)

    def taint_type(self, **kwargs):
        if self.closed:
            raise ClosedGraphError("Unable to taint type", node=self)
        self.fd.taint_type(**kwargs)

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

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        raise DagflowError("Unimplemented method: the method must be overridden!")

    def _on_taint(self, caller: Input | None):
        """A node method to be called on taint"""
        pass

    def _post_allocate(self):
        self._input_nodes_callbacks = []

        for input in self.inputs.iter_all():
            node = input.parent_node
            if node not in self._input_nodes_callbacks:
                self._input_nodes_callbacks.append(node.touch)

    def update_types(self, recursive: bool = True):
        if not self.fd.types_tainted:
            return True
        # TODO: causes problems with nodes, that are allocated and closed prior the graph being closed
        # Need a mechanism to request reallocation
        if recursive:
            self.logger.debug(f"Node '{self.name}': Trigger recursive update types...")
            for input in self.inputs.iter_all():
                if not input.connected():
                    raise ClosingError("Input is not connected", node=self, input=input)
                input.parent_node.update_types(recursive)
        self.logger.debug(f"Node '{self.name}': Update types...")
        self._typefunc()
        self.fd.types_tainted = False

        self._fd.needs_reallocation = True

    def allocate(self, recursive: bool = True):
        if self._fd.allocated and not self._fd.needs_reallocation:
            return True
        if recursive:
            self.logger.debug(f"Node '{self.name}': Trigger recursive memory allocation...")
            for _input in self.inputs.iter_all():
                try:
                    parent_node = _input.parent_node
                except AttributeError as exc:
                    raise ClosingError("Parent node is not initialized", input=_input) from exc
                if not parent_node.allocate(recursive):
                    return False
        self.logger.debug(f"Node '{self.name}': Allocate memory on inputs")
        input_reassigned = self.inputs.allocate()
        self.logger.debug(f"Node '{self.name}': Allocate memory on outputs")
        output_reassigned = self.outputs.allocate()
        self.logger.debug(f"Node '{self.name}': Post allocate")
        if input_reassigned or output_reassigned or self._fd.needs_post_allocate:
            self._post_allocate()
        self._fd.allocated = True
        self._fd.needs_reallocation = False
        return True

    def close(
        self,
        recursive: bool = True,
        strict: bool = True,
        close_children=False,
        together: Sequence["Node"] = [],
    ) -> bool:
        # Caution: `together` list should not be written in!

        if self.closed:
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
        self.fd.closed = self.fd.allocated
        if strict and not self.closed:
            raise ClosingError(node=self)

        if close_children:
            for output in self.outputs:
                for input in output.child_inputs:
                    input.node.close(close_children=True, strict=strict)

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
        self.fd.closed = False
        return not self.closed
