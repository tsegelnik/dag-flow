from __future__ import annotations

from timeit import repeat
from collections.abc import Sequence
from textwrap import shorten

from pandas import DataFrame, Series, concat
import numpy

from .profiler import Profiler
from dagflow.nodes import FunctionNode

SOURCE_COL_WIDTH = 32
SINK_COL_WIDTH = 32

# it is possible to group by two columns
_ALLOWED_GROUPBY = (
    ["source nodes", "sink nodes"],
    "source nodes",
    "sink nodes",
)

class FrameworkProfiler(Profiler):
    """Profiler class that used to estimate
    the interaction time between nodes (framework time)"""
    __slots__ = ()

    def __init__(self,
                 target_nodes: Sequence[FunctionNode]=[],
                 *,
                 sources: Sequence[FunctionNode]=[],
                 sinks: Sequence[FunctionNode]=[],
                 n_runs = 100) -> None:
        super().__init__(target_nodes, sources, sinks, n_runs)
        self._allowed_groupby = _ALLOWED_GROUPBY
        self.register_agg_func(
            func=self._t_single_node,
            aliases=['t_single_by_node', 'single_by_node',
                     'mean_by_node', 't_mean_by_node'],
            column_name='t_single_by_node')
        if not (self._sources and self._sinks):
            self._reveal_source_sink()

    def _t_single_node(self, _s: Series) -> Series:
        """Return mean framework time normilized by one node.
        This as also an example of user-defined aggregate function
        """
        nodes_count = len(self._target_nodes)
        return Series({'t_single_by_node': numpy.mean(_s) / nodes_count})

    def _taint_nodes(self):
        for node in self._target_nodes:
            node.taint()

    @staticmethod
    def fcn_no_computation(node: FunctionNode):
        for input in node.inputs.iter_all():
            input.touch()

    def _make_fcns_empty(self):
        for node in self._target_nodes:
            node._stash_fcn()
            # __get__ - the way to bind method to an instance
            node.fcn = self.fcn_no_computation.__get__(node)

    def _restore_fcns(self):
        for node in self._target_nodes:
            node._unwrap_fcn()

    def _estimate_framework_time(self) -> list[float]:
        self._make_fcns_empty()
        def repeat_stmt():
            for sink_node in self._sinks:
                sink_node.eval()
        repeat_stmt()   # touch all dependent nodes before estimations
        results = repeat(stmt=repeat_stmt, setup=self._taint_nodes,
                         repeat=self._n_runs, number=1)
        self._restore_fcns()
        self._taint_nodes()
        return results

    def _shorten_names(self, nodes, max_length):
        names = []
        names_sum_length = 0
        for node in nodes:
            if names_sum_length > max_length:
                break
            names.append(node.name)
            names_sum_length += len(node.name)
        return shorten( ", ".join(names) , max_length)

    def estimate_framework_time(self,
                                append_results: bool=False) -> FrameworkProfiler:
        results = self._estimate_framework_time()
        df = DataFrame(results, columns=["time"])
        sinks_short = self._shorten_names(self._sinks, SINK_COL_WIDTH)
        sources_short = self._shorten_names(self._sources, SOURCE_COL_WIDTH)
        df.insert(0, "sink nodes", sinks_short)
        df.insert(0, "source nodes", sources_short)
        if append_results and hasattr(self, "_estimations_table"):
            self._estimations_table = concat([self._estimations_table, df])
        else:
            self._estimations_table = df
        return self

    def make_report(self,
                    group_by=["source nodes", "sink nodes"],
                    agg_funcs: Sequence[str] | None=None,
                    sort_by: str | None=None):
        return super().make_report(group_by, agg_funcs, sort_by)

    def print_report(self,
                     rows: int | None=10,
                     group_by=["source nodes", "sink nodes"],
                     agg_funcs: Sequence[str] | None=None,
                     sort_by: str | None=None) -> DataFrame:
        report = self.make_report(group_by, agg_funcs, sort_by)
        print(f"\nFramework Profling {hex(id(self))}, "
              f"n_runs for given subgraph: {self._n_runs}, "
              f"nodes in subgraph: {len(self._target_nodes)}\n"
              f"sort by: `{sort_by or 'default sorting'}`, "
              f"max rows displayed: {rows}")
        super()._print_table(report, rows)
        return report
