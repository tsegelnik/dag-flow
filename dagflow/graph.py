from __future__ import annotations

from .exception import ClosedGraphError, ClosingError, InitializationError, UnclosedGraphError
from .graphbase import GraphBase
from .logger import Logger, get_logger


class Graph(GraphBase):
    """
    The graph class:
    holds nodes as a list, has name, label, logger and uses context
    """

    __slots__ = (
        "_label",
        "_name",
        "_close",
        "_strict",
        "_closed",
        "_debug",
        "_logger",
    )

    _label: str | None
    _name: str
    _close: bool
    _closed: bool
    _debug: bool
    _logger: Logger

    def __init__(self, *args, close: bool = False, strict: bool = True, **kwargs):
        super().__init__(*args)
        self._label = kwargs.pop("label", None)
        self._name = kwargs.pop("name", "graph")
        self._debug = kwargs.pop("debug", False)
        self._close = close
        self._strict = strict
        self._closed = False
        # init or get default logger
        self._logger = get_logger(
            filename=kwargs.pop("logfile", None),
            debug=self.debug,
            console=kwargs.pop("console", True),
            formatstr=kwargs.pop("logformat", None),
            name=kwargs.pop("loggername", None),
        )
        if kwargs:
            raise InitializationError(f"Unparsed arguments: {kwargs}!")

    @property
    def debug(self) -> bool:
        return self._debug

    @property
    def logger(self) -> Logger:
        return self._logger

    @property
    def name(self) -> str:
        return self._name

    @property
    def closed(self) -> bool:
        return self._closed

    def _add_output(self, *args, **kwargs):
        """Dummy method"""

    def _add_input(self, *args, **kwargs):
        """Dummy method"""

    def label(self):
        """Returns formatted label"""
        if self._label:
            return self._label.format(self._label, nodes=len(self._nodes))

    def add_node(self, name, **kwargs):
        """
        Adds a node, if the graph is opened.
        It is possible to pass the node class via the `nodeclass` arg
        (default: `Node`)
        """
        if not self.closed:
            from .node import Node

            return kwargs.pop("nodeclass", Node)(name, graph=self, **kwargs)
        raise ClosedGraphError(node=name)

    def add_nodes(self, nodes, **kwargs):
        """Adds nodes"""
        if not self.closed:
            return (self.add_node(node, **kwargs) for node in nodes)
        raise ClosedGraphError(node=nodes)

    def print(self):
        print(f"Graph with {len(self._nodes)} nodes")
        for node in self._nodes:
            node.print()

    def close(self, *, strict: bool = True, **kwargs) -> bool:
        """Closes the graph"""
        if self._closed:
            return True
        self.logger.debug(f"Graph '{self.name}': Closing...")
        self.logger.debug(f"Graph '{self.name}': Update types...")
        for node in self._nodes:
            try:
                node.update_types()
            except ClosingError:
                if strict:
                    raise
        self.logger.debug(f"Graph '{self.name}': Allocate memory...")
        for node in self._nodes:
            try:
                node.allocate(**kwargs)
            except ClosingError:
                if strict:
                    raise
        self.logger.debug(f"Graph '{self.name}': Closing nodes...")
        for node in self._nodes:
            try:
                self._closed = node.close(**kwargs)
            except ClosingError:
                if strict:
                    raise
            if not self._closed:
                break
        else:
            self._closed = True
        if strict and not self._closed:
            raise UnclosedGraphError("The graph is still open!")
        self.logger.debug(
            f"Graph '{self.name}': The graph {self._closed and 'is closed' or 'failed to close'}!"
        )
        return self._closed

    def open(self, force: bool = False) -> bool:
        """Opens the graph recursively"""
        if not self._closed and not force:
            return True
        self.logger.debug(f"Graph '{self.name}': Opening...")
        self._closed = not all(node.open(force) for node in self._nodes)
        if self._closed:
            raise UnclosedGraphError("The graph is still open!")
        return not self._closed

    def build_index_dict(self, index):
        for node in self:
            node.labels.build_index_dict(index)

            for output in node.outputs.iter_all():
                output.labels.build_index_dict(index)

    @classmethod
    def current(cls) -> Graph | None:
        return _context_graph[-1] if _context_graph else None

    def __enter__(self):
        _context_graph.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if _context_graph.pop() != self:
            raise RuntimeError("Graph: invalid context exit")

        if exc_val is not None:
            raise exc_val

        if self._close:
            self.close(strict=self._strict)


_context_graph: list[Graph] = []
