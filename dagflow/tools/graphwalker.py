from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from typing import Callable, TYPE_CHECKING, Generator

from ordered_set import OrderedSet

from ..node import Node, Output

if TYPE_CHECKING:
    from ..graph import Graph



class GraphWalker:
    """Simple Graph walker.

    TODO: import features from graphviz
    """
    __slots__ = (
        "_include_only",
        "_initial_nodes",
        "_queue",
        "_skipped_nodes",
        "_cache_nodes",
        "_cache_inputs",
        "_cache_outputs",
        "_cache_inputs_open",
        "_cache_outputs_open",
    )

    _include_only: tuple[Node, ...]
    _initial_nodes: list

    _queue: deque[Node]
    _skipped_nodes: set[Node]
    _cache_nodes: OrderedSet
    _cache_inputs: OrderedSet
    _cache_outputs: OrderedSet
    _cache_inputs_open: OrderedSet
    _cache_outputs_open: OrderedSet

    def __init__(
        self, *args: Node | Output | Iterable[Node] | Iterable[Output], *, include_only: Iterable[Node]
    ):
        self._include_only = tuple(include_only)

        self._initial_nodes = []
        for arg in args:
            self._add_initial_node(arg)

        self._queue = deque()
        self._skipped_nodes = set()
        self._cache_nodes = OrderedSet()
        self._cache_inputs = OrderedSet()
        self._cache_outputs = OrderedSet()
        self._cache_inputs_open = OrderedSet()
        self._cache_outputs_open = OrderedSet()

        self._build_cache()

    @classmethod
    def from_graph(cls, graph: Graph, **kwargs) -> GraphWalker:
        node = graph._nodes[0]
        return cls(node, **kwargs)

    def _add_initial_node(self, arg: Node | Output | Iterable[Node] | Iterable[Output]):
        match arg:
            case Sequence():
                args = arg
            case Iterable():
                args = list(arg)  # pyright: ignore [reportAssignmentType]
            case _:
                args = [arg]  # pyright: ignore [reportAssignmentType]

        for item in args:
            match item:
                case Output():
                    self._initial_nodes.append(item.node)
                case Node():
                    self._initial_nodes.append(item)
                case _:
                    raise ValueError()

    def _build_cache(self):
        self._add_to_queue(*self._initial_nodes)

        while self._queue:
            node = self._queue.popleft()

            skip = self._if_skip_node(node)
            if skip:
                self._skipped_nodes.add(node)
            else:
                self._cache_nodes.append(node)
                self._cache_inputs.update(node.inputs.iter_all())
                self._cache_outputs.update(node.outputs.iter_all())

            self._propagate_forward(node, skip)
            self._propagate_backward(node, skip)

    def _add_to_queue(self, *nodes: Node):
        for node in nodes:
            if node in self._queue or node in self._cache_nodes or node in self._skipped_nodes:
                continue

            self._queue.append(node)

    def _propagate_forward(self, node: Node, skip: bool):
        for output in node.outputs.iter_all():
            for child_input in output.child_inputs:
                self._add_to_queue(child_input.node)

            if skip:
                continue

            if not output.child_inputs:
                self._cache_outputs_open.add(output)

    def _propagate_backward(self, node: Node, skip: bool):
        for input in node.inputs.iter_all():
            output = input.parent_output

            if output:
                self._add_to_queue(output.node)
            elif not skip:
                self._cache_inputs_open.append(input)

    def _if_skip_node(self, node: Node) -> bool:
        if not self._include_only:
            return False

        # label = node.labels.text

        # for p in self._include_only:
        #     if p in label:
        #         return False

        return True

    def _list_do(self, seq: Sequence, *functions: Callable):
        for obj in seq:
            for fcn in functions:
                fcn(obj)

    def nodes(self) -> Generator[Node]:
        yield from self._cache_nodes

    def node_do(self, *args: Callable):
        return self._list_do(self._cache_nodes, *args)

    def output_do(self, *args: Callable):
        return self._list_do(self._cache_outputs, *args)

    def input_do(self, *args: Callable):
        return self._list_do(self._cache_inputs, *args)

    def input_open_do(self, *args: Callable):
        return self._list_do(self._cache_inputs_open, *args)

    def output_open_do(self, *args: Callable):
        return self._list_do(self._cache_outputs_open, *args)

