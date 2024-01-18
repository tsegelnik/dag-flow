from __future__ import annotations

from timeit import timeit
from collections.abc import Sequence

import pandas as pd

from .profiling import Profiling
from dagflow.nodes import FunctionNode


class IndividualProfiling(Profiling):
    """Profiling class for estimating the time of individual nodes"""
    _n_runs: int
    _estimations_table: pd.DataFrame

    DEFAULT_RUNS = 10000
    _TABLE_COLUMNS = ("node", "type", "name", "time")
    _ALLOWED_GROUPBY = ("node", "type", "name")

    def __init__(self,
                 target_nodes: Sequence[FunctionNode]=[],
                 *,
                 source: Sequence[FunctionNode]=[],
                 sink: Sequence[FunctionNode]=[],
                 n_runs: int=DEFAULT_RUNS):
        super().__init__(target_nodes, source, sink, n_runs)

    @classmethod
    def estimate_node(cls, node: FunctionNode, n_runs: int=DEFAULT_RUNS):
        for input in node.inputs.iter_all():
            input.touch()
        return timeit(stmt=node.fcn, number=n_runs)

    def estimate_target_nodes(self) -> IndividualProfiling:
        records = {col: [] for col in self._TABLE_COLUMNS}
        for node in self._target_nodes:
            estimations = self.estimate_node(node, self._n_runs)
            records["node"].append(node)
            records["type"].append(type(node).__name__)
            records["name"].append(node.name)
            records["time"].append(estimations)
        self._estimations_table = pd.DataFrame(records)
        return self

    def make_report(self,
                    group_by: str | None="type",
                    agg_funcs: Sequence[str] | None=None,
                    sort_by: str | None=None,
                    normilize=True):
        report = super().make_report(group_by, agg_funcs, sort_by)
        if normilize:
            return self._normalize(report)
        return report

    def _print_total_time(self):
        total = self._estimations_table['time'].sum()
        print("total estimations time"
              " / n_runs: %.9f sec." % (total / self._n_runs))
        print("total estimations time: %.6f sec." % total)

    def print_report(self,
                     rows: int | None=10,
                     group_by: str | None="type",
                     agg_funcs: Sequence[str] | None=None,
                     sort_by: str | None=None):
        report = self.make_report(group_by, agg_funcs, sort_by)
        print(f"\nIndividual Profilng {hex(id(self))}, "
              f"n_runs for each node: {self._n_runs}\n"
              f"sort by: {sort_by or 'default sorting'}, "
              f"max rows displayed: {rows}")
        super()._print_table(report, rows)
        self._print_total_time()
