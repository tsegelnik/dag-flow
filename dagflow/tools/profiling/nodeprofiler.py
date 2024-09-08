from __future__ import annotations

from typing import TYPE_CHECKING
from timeit import timeit
from collections.abc import Sequence

from pandas import DataFrame

from .timerprofiler import TimerProfiler
if TYPE_CHECKING:
    from dagflow.node import Node

_ALLOWED_GROUPBY = ("node", "type", "name")

class NodeProfiler(TimerProfiler):
    """Profiling class for estimating the time of individual nodes"""
    __slots__ = ()

    def __init__(
        self,
        target_nodes: Sequence[Node] = (),
        *,
        sources: Sequence[Node] = (),
        sinks: Sequence[Node] = (),
        n_runs: int = 10_000
    ):
        super().__init__(target_nodes, sources, sinks, n_runs)
        self._allowed_groupby = _ALLOWED_GROUPBY
        self._primary_col = "time"

    @classmethod
    def estimate_node(cls, node: Node, n_runs: int = 10_000):
        for input in node.inputs.iter_all():
            input.touch()
        return timeit(stmt=node.fcn, number=n_runs)

    def estimate_target_nodes(self) -> NodeProfiler:
        records = {col: [] for col in ("node", "type", "name", "time")}
        for node in self._target_nodes:
            estimations = self.estimate_node(node, self._n_runs)
            records["node"].append(str(node))
            records["type"].append(type(node).__name__)
            records["name"].append(node.name)
            records["time"].append(estimations)
        self._estimations_table = DataFrame(records)
        return self

    def make_report(
        self,
        group_by: str | tuple[str] | None = "type",
        agg_funcs: Sequence[str] | None = None,
        sort_by: str | None = None,
        normilize: bool = True
    ) -> DataFrame:
        report = super().make_report(group_by, agg_funcs, sort_by)
        if normilize:
            return self._normalize(report)
        return report

    def _print_total_time(self):
        total = self._total_estimations_time()
        print("total estimations time"
              " / n_runs: %.9f sec." % (total / self._n_runs))
        print("total estimations time: %.6f sec." % total)

    def print_report(
        self,
        rows: int | None = 40,
        group_by: str | None = "type",
        agg_funcs: Sequence[str] | None = None,
        sort_by: str | None = None
    ) -> DataFrame:
        report = self.make_report(group_by, agg_funcs, sort_by)
        print(f"\nNode Profiling {hex(id(self))}, "
              f"n_runs for each node: {self._n_runs}\n"
              f"sort by: `{sort_by or 'default sorting'}`, "
              f"group by: `{group_by or 'no grouping'}`")
        super()._print_table(report, rows)
        self._print_total_time()
        return report
