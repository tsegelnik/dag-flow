from __future__ import annotations

from timeit import repeat
from collections.abc import Sequence
from textwrap import shorten
import types

import pandas as pd

from .profiling import Profiling
from dagflow.nodes import FunctionNode

SOURCE_COL_WIDTH = 32
SINK_COL_WIDTH = 32

class FrameworkProfiling(Profiling):
    _ALLOWED_GROUPBY = [["source nodes", "sink nodes"],  # [a, b] - group by two
                       "source nodes",                   #  columns a and b
                       "sink nodes"]

    def __init__(self,
                 target_nodes: Sequence[FunctionNode]=[],
                 *,
                 source: Sequence[FunctionNode]=[],
                 sink: Sequence[FunctionNode]=[],
                 n_runs = 100) -> None:
        super().__init__(target_nodes, source, sink, n_runs)
        if not self._source or not self._sink:
            self._reveal_source_sink()

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
            node.fcn = types.MethodType(self.fcn_no_computation, node)

    def _restore_fcns(self):
        for node in self._target_nodes:
            node._unwrap_fcn()

    def _estimate_framework_time(self) -> list[float]:
        self._make_fcns_empty()
        setup = lambda: self._taint_nodes()
        def repeat_stmt():
            for sink_node in self._sink:
                sink_node.eval()
        results = repeat(stmt=repeat_stmt, setup=setup,
                         repeat=self._n_runs, number=1)
        self._restore_fcns()
        return results

    def estimate_framework_time(self,
                                append_results: bool=False) -> FrameworkProfiling:
        results = self._estimate_framework_time()
        df = pd.DataFrame(results, columns=["time"])
        sink_names = str([n.name for n in self._sink])
        source_names = str([n.name for n in self._source])
        df.insert(0, "sink nodes", shorten(sink_names, width=SINK_COL_WIDTH))
        df.insert(0, "source nodes", shorten(source_names, width=SOURCE_COL_WIDTH))
        if append_results and hasattr(self, "_estimations_table"):
            self._estimations_table = pd.concat([self._estimations_table, df])
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
                     sort_by: str | None=None):
        report = self.make_report(group_by, agg_funcs, sort_by)
        print(f"\nFramework Profling {hex(id(self))}, "
              f"n_runs for given subgraph: {self._n_runs}, "
              f"nodes in subgraph: {len(self._target_nodes)}\n"
              f"sort by: `{sort_by or 'default sorting'}`, "
              f"max rows displayed: {rows}")
        super()._print_table(report, rows)
