from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
from .legs import Legs
from .logger import Logger, get_logger
from .output import Output
from .types import GraphT
from typing import Optional, List, Dict, Union, Callable, Any, Tuple, Generator

class Node(Legs):
    _name: str
    _mark: Optional[str] = None
    _labels: Dict[str, str]
    _graph: Optional[GraphT] = None
    _fcn: Optional[Callable] = None
    _fcn_chain = None
    _exception: Optional[str] = None

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

    def __init__(
        self,
        name,
        *,
        label: Union[str, dict, None] = None,
        graph: Optional[GraphT] = None,
        fcn: Optional[Callable] = None,
        typefunc: Optional[Callable] = None,
        debug: Optional[bool] = None,
        logger: Optional[Any] = None,
        missing_input_handler: Optional[Callable] = None,
        immediate: bool = False,
        auto_freeze: bool = False,
        frozen: bool = False,
        **kwargs,
    ):
        super().__init__(missing_input_handler=missing_input_handler)
        self._name = name
        if fcn is not None:
            self._fcn = fcn
        if typefunc is not None:
            self._typefunc = typefunc
        elif typefunc is False:
            self._typefunc = lambda: None

        self._fcn_chain = []
        if graph is None:
            from .graph import Graph

            self.graph = Graph.current()
        else:
            self.graph = graph

        if debug is None and self.graph is not None:
            self._debug = self.graph.debug
        else:
            self._debug = bool(debug)

        if isinstance(label, str):
            self._labels = {'text': label}
        elif isinstance(label, dict):
            self._labels = label
        else:
            self._labels = {'text': name}

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
    def mark(self):
        return self._mark

    @property
    def exception(self):
        return self._exception

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
        elif any(input.invalid for input in self.inputs.iter_all()):
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
        for input in self.inputs.iter_all():
            node = input.parent_node
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
    def labels(self) -> Generator[Tuple[str,str], None, None]:
        yield from self._labels.items()

    #
    # Methods
    #
    def __call__(self, name, child_output: Optional[Output] = None, **kwargs):
        self.logger.debug(f"Node '{self.name}': Get input '{name}'")
        kwargs.setdefault("positional", False)
        inp = self.inputs.get(name, None)
        if inp is None:
            if self.closed:
                raise ClosedGraphError(node=self)
            return self._add_input(name, child_output=child_output, **kwargs)
        elif inp.connected and (output := inp.parent_output):
            raise ReconnectionError(input=inp, node=self, output=output)
        return inp

    def label(self, source='text'):
        # if self._labels:
        #     kwargs.setdefault("name", self._name)
        #     return self._labels.format(*args, **kwargs)
        label = self._labels.get(source, None)
        if label is None:
            return self._labels['text']

        return label

    def _inherit_labels(self, source: 'Node', fmt: Union[str, Callable]) -> None:
        if isinstance(fmt, str):
            formatter = fmt.format
        elif isinstance(fmt, dict):
            formatter = lambda s: fmt.get(s, s)
        else:
            formatter = fmt

        for k, v in source.labels:
            if k in ('key',):
                continue
            newv = formatter(v)
            if newv is not None:
                self._labels[k] = newv

    def add_input(self, name, **kwargs) -> Union[Input, Tuple[Input]]:
        if not self.closed:
            return self._add_input(name, **kwargs)
        raise ClosedGraphError(node=self)

    def _add_input(self, name, **kwargs) -> Union[Input, Tuple[Input]]:
        if IsIterable(name):
            return tuple(self._add_input(n, **kwargs) for n in name)
        self.logger.debug(f"Node '{self.name}': Add input '{name}'")
        if name in self.inputs:
            raise ReconnectionError(input=name, node=self)
        positional = kwargs.pop("positional", True)
        keyword = kwargs.pop("keyword", True)
        inp = Input(name, self, **kwargs)
        self.inputs.add(inp, positional=positional, keyword=keyword)

        if self._graph:
            self._graph._add_input(inp)
        return inp

    def add_output(self, name, **kwargs) -> Union[Output, Tuple[Output]]:
        if not self.closed:
            return self._add_output(name, **kwargs)
        raise ClosedGraphError(node=self)

    def _add_output(
        self, name, *, keyword: bool = True, positional: bool = True, **kwargs
    ) -> Union[Output, Tuple[Output]]:
        if IsIterable(name):
            return tuple(self._add_output(n, **kwargs) for n in name)
        self.logger.debug(f"Node '{self.name}': Add output '{name}'")
        if isinstance(name, Output):
            if name.name in self.outputs or name.node:
                raise ReconnectionError(output=name, node=self)
            name._node = self
            return self.__add_output(
                name, positional=positional, keyword=keyword
            )
        if name in self.outputs:
            raise ReconnectionError(output=name, node=self)

        return self.__add_output(
            Output(name, self, **kwargs),
            positional=positional,
            keyword=keyword,
        )

    def __add_output(
        self, out, positional: bool = True, keyword: bool = True
    ) -> Union[Output, Tuple[Output]]:
        self.outputs.add(out, positional=positional, keyword=keyword)
        if self._graph:
            self._graph._add_output(out)
        return out

    def add_pair(
        self, iname: str, oname: str, **kwargs
    ) -> Tuple[Input, Output]:
        if not self.closed:
            return self._add_pair(iname, oname, **kwargs)
        raise ClosedGraphError(node=self)

    def _add_pair(
        self,
        iname: str,
        oname: str,
        input_kws: Optional[dict] = None,
        output_kws: Optional[dict] = None,
    ) -> Tuple[Input, Output]:
        input_kws = input_kws or {}
        output_kws = output_kws or {}
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

    def _fcn(self, _, inputs, outputs):
        pass

    def _on_taint(self, caller: Input):
        """A node method to be called on taint"""
        pass

    def _post_allocate(self):
        pass

    def update_types(self, recursive: bool = True) -> bool:
        if not self._types_tainted:
            return True
        if recursive:
            self.logger.debug(
                f"Node '{self.name}': Trigger recursive update types..."
            )
            for input in self.inputs.iter_all():
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
                input.parent_node.allocate(recursive)
                for input in self.inputs.iter_all()
            ):
                return False
        self.logger.debug(f"Node '{self.name}': Allocate memory on inputs")
        if not self.inputs.allocate():
            raise AllocationError(
                "Cannot allocate memory for inputs!", node=self
            )
        self.logger.debug(f"Node '{self.name}': Allocate memory on outputs")
        if not self.outputs.allocate():
            raise AllocationError(
                "Cannot allocate memory for outputs!", node=self
            )
        self.logger.debug(f"Node '{self.name}': Post allocate")
        self._post_allocate()
        self._allocated = True
        return True

    def close(
        self, recursive: bool = True, together: List["Node"] = []
    ) -> bool:
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
            input.parent_node.close(recursive)
            for input in self.inputs.iter_all()
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
    def get_data(self, key=0):
        return self.outputs[key].data

    def get_input_data(self, key):
        return self.inputs[key].data()

    def to_dict(self, *, label_from: str='text') -> dict:
        data = self.get_data()
        if data.size>1:
            raise AttributeError('to_dict')
        return {
                'value': data[0],
                'label': self.label(label_from)
                }
