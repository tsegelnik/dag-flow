
from .logger import Logger, get_logger
from .node_group import NodeGroup
from .tools import undefined


class Graph(NodeGroup):
    """
    The graph class:
    holds nodes as a list, has name, label, logger and uses context
    """

    _context_graph = undefined("graph")
    _label = undefined("label")
    _name = "graph"
    _closed: bool = False
    _debug: bool = False
    _logger: Logger

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self._label = kwargs.pop("label", undefined("label"))
        self._name = kwargs.pop("name", "graph")
        self._debug = kwargs.pop("debug", False)
        # init or get default logger
        self._logger = get_logger(
            filename=kwargs.pop("logfile", None),
            debug=self.debug,
            console=kwargs.pop("console", True),
            formatstr=kwargs.pop("logformat", None),
            name=kwargs.pop("loggername", None),
        )
        if kwargs:
            raise RuntimeError(f"Unparsed arguments: {kwargs}!")

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
        pass

    def _add_input(self, *args, **kwargs):
        """Dummy method"""
        pass

    def label(self):
        """Returns formatted label"""
        if self._label:
            return self._label.format(self._label, nodes=len(self._nodes))

    def add_node(self, name, **kwargs):
        """
        Adds a node, if the graph is opened.
        It is possible to pass the node class via the `nodeclass` arg
        (default: `FunctionNode`)
        """
        from .nodes import FunctionNode
        if not self.closed:
            return kwargs.pop("nodeclass", FunctionNode)(
                name, graph=self, **kwargs
            )
        self.logger.warning(
            f"Graph '{self.name}': "
            "A modification of the closed graph is restricted!"
        )

    def add_nodes(self, nodes, **kwargs):
        """Adds nodes"""
        if not self.closed:
            return (self.add_node(node, **kwargs) for node in nodes)
        self.logger.warning(
            f"Graph '{self.name}': "
            "A modification of the closed graph is restricted!"
        )

    def print(self):
        print(f"Graph with {len(self._nodes)} nodes")
        for node in self._nodes:
            node.print()

    @classmethod
    def current(cls):
        return cls._context_graph

    def __enter__(self):
        Graph._context_graph = self
        return self

    def __exit__(self, *args, **kwargs):
        Graph._context_graph = undefined("graph")

    def close(self, **kwargs) -> bool:
        """Closes the graph"""
        self.logger.debug(f"Graph '{self.name}': Closing...")
        if self._closed:
            return True
        self.logger.debug(f"Graph '{self.name}': Update types...")
        try:
            for node in self._nodes:
                node.update_types(**kwargs)
        except Exception as exc:
            self.logger.warning(
                f"Graph '{self.name}': Exception occured during type "
                f"updating: {exc}!"
            )
            self._closed = False
            return self._closed
        self.logger.debug(f"Graph '{self.name}': Allocate memory...")
        try:
            for node in self._nodes:
                node.allocate(**kwargs)
        except Exception as exc:
            self.logger.warning(
                f"Graph '{self.name}': Some nodes are not allocated: "
                f"'{tuple(node.name for node in self._nodes if not node.allocated)}'!"
                f"Catched exception: {exc}"
            )
            self._closed = False
            return self._closed
        self.logger.debug(f"Graph '{self.name}': Close nodes...")
        self._closed = all(node._close(**kwargs) for node in self._nodes)
        if not self._closed:
            self.logger.warning(
                f"Graph '{self.name}': Some nodes are not closed: "
                f"'{tuple(node.name for node in self._nodes if not node.closed)}'!"
            )
        else:
            self.logger.debug(
                f"Graph '{self.name}': The graph is closed successfully."
            )
        return self._closed

    # def close(self, **kwargs) -> bool:
    #    """Closes the graph recursively"""
    #    self.logger.debug(f"Graph '{self.name}': Closing...")
    #    if self._closed:
    #        return True
    #    self._closed = all(node.close(**kwargs) for node in self._nodes)
    #    if not self._closed:
    #        self.logger.warning(
    #            f"Graph '{self.name}': Some nodes are still open: "
    #            f"'{tuple(node.name for node in self._nodes if not node.closed)}'!"
    #        )
    #    return self._closed

    def open(self) -> bool:
        """Opens the graph recursively"""
        self.logger.debug(f"Graph '{self.name}': Opening...")
        if not self._closed:
            return True
        self._closed = not all(node.open() for node in self._nodes)
        if self._closed:
            self.logger.warning(
                f"Graph '{self.name}': Some nodes are still open: "
                f"'{tuple(node.name for node in self._nodes if node.closed)}'!"
            )
        return not self._closed
