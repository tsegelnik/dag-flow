from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pandas import DataFrame

if TYPE_CHECKING:
    from typing import Sequence
    from dagflow.node import Node
    from dagflow.input import Input
    from dagflow.output import Output

from .profiler import Profiler


class MemoryProfiler(Profiler):
    def __init__(
        self,
        target_nodes: Sequence[Node]=(),
        *,
        sources: Sequence[Node]=(),
        sinks: Sequence[Node]=()
    ):
        super().__init__(target_nodes, sources, sinks, n_runs=1)
        self._default_sort_col = 'size'

    def _touch_nodes(self):
        for node in self._target_nodes:
            node.eval()

    @classmethod
    def estimate_node(cls, node) -> dict[Input | Output, int]:
        """Return `dict` of sizes for each Input/Output of given `node`"""
        estimations = {}
        inp: Input
        out: Output
        for inp in node.inputs.iter_all():
            if inp.has_data and inp.owns_buffer:
                estimations[inp] = inp._own_data.nbytes
            else:
                estimations[inp] = 0
        for out in node.outputs.iter_all():
            if (out.has_data 
                and (out.owns_buffer or out._allocating_input is None)):
                # If there is an _allocating_input, 
                #  the `out.data` refers to the child `Input` data.
                # However if there is no `_allocating_input` 
                #  and owns_buffer=False (and `out.data` is not `None` of course)
                #  then it means there is allocated memory for this Output.
                estimations[out] = out.data_unsafe.nbytes
            else:
                estimations[out] = 0
        return estimations


    def estimate_target_nodes(self, touch=True):
        """Estimates size of all edges of all `self.target_nodes`.
        
        Return current `MemoryProfiler` instance.
        """
        data_sizes = {}
        if touch:
            self._touch_nodes()
        
        records = {col: [] for col in ("node", "type", "edge", "size")}
        for node in self._target_nodes:
            estimations = self.estimate_node(node)
            for edge, size in estimations.items():
                records["node"].append(node)
                records["type"].append(type(node).__name__)
                records["edge"].append(edge)
                records["size"].append(size)
        self._estimations_table = DataFrame(records)
        return self
    @property
    def total_size(self):
        """Return size of all edges of '_target_nodes' in bytes
        """
        if not hasattr(self, '_estimations_table'):
            self.estimate_target_nodes()
        return self._estimations_table["size"].sum()

    def make_report(
        self,
        group_by: str | list[str] | None,
        agg_funcs: Sequence[str] | None,
        sort_by: str | None
    ) -> DataFrame:
        raise NotImplementedError
        return super().make_report(group_by, agg_funcs, sort_by)
    
    def print_report(
        self,
        rows: int | None,
        group_by: str | list[str] | None,
        agg_funcs: Sequence[str] | None,
        sort_by: str | None
    ) -> DataFrame:
        raise NotImplementedError
        return super().print_report(rows, group_by, agg_funcs, sort_by)