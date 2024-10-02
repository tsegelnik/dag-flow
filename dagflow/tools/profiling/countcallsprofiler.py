from __future__ import annotations

from typing import TYPE_CHECKING

from pandas import DataFrame

from .profiler import Profiler
if TYPE_CHECKING:
    from dagflow.node import Node
    from collections.abc import Callable
    from collections.abc import Sequence

# columnt aliases for aggregate functions
_COLUMN_ALIASES: dict[str | Callable, str] = {
    "mean": "single",
    "count": "node_count",
}

# The same names are needed to show possible aggregation functions in case of error
_AGG_ALIASES: dict[str, str | Callable] = {
    "single": "mean",
    "mean": "mean",
    "median": "median",
    "sum": "sum",
    "std": "std",
    "node_count": "count",
    "min": "min",
    "max": "max",
    "var": "var",
}

class CountCallsProfiler(Profiler):
    """Profiling class for estimating number of calls
    of each node after model fit"""
    __slots__ = ()

    def __init__(
        self,
        target_nodes: Sequence[Node] = (),
        *,
        sources: Sequence[Node] = (),
        sinks: Sequence[Node] = (),
    ):
        super().__init__(target_nodes, sources, sinks, n_runs=1)
        self._primary_col = "calls"
        self._default_agg_funcs = ("count", "mean", "sum")
        self._agg_aliases = _AGG_ALIASES.copy()
        self._column_aliases = _COLUMN_ALIASES.copy()

    def estimate_calls(self) -> CountCallsProfiler:
        records = {col: [] for col in ("node", "type", "name", "calls")}
        for node in self._target_nodes:
            records["node"].append(str(node))
            records["type"].append(type(node).__name__)
            records["name"].append(node.name)
            records["calls"].append(node.n_calls)
        self._estimations_table = DataFrame(records)
        return self

    def make_report(
        self,
        group_by: str | list[str] | None = "type",
        agg_funcs: Sequence[str] | None = None,
        sort_by: str | None = None
    ) -> DataFrame:
        return super().make_report(group_by, agg_funcs, sort_by)

    def calls_by_node(self):
        if not hasattr(self, '_estimations_table'):
            self.estimate_calls()
        count =  len(self._target_nodes)
        calls = self._estimations_table["calls"].sum()
        return calls / count

    def print_report(
        self,
        rows: int | None = 40,
        group_by: str | list[str] | None = "type",
        agg_funcs: Sequence[str] | None = None,
        sort_by: str | None = None
    ) -> DataFrame:
        report = self.make_report(group_by, agg_funcs, sort_by)
        print(f"\nCounts of calls profiling {hex(id(self))}, "
              f"sort by: `{sort_by or 'default sorting'}`, "
              f"group by: `{group_by or 'no grouping'}`")
        super()._print_table(report, rows)
        print(f"Mean count of calls by node: \t{self.calls_by_node():.2f}")
        return report
