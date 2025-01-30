from __future__ import annotations

from timeit import timeit
from typing import TYPE_CHECKING

from pandas import DataFrame

from .timer_profiler import TimerProfiler

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dagflow.core.node import Node

_ALLOWED_GROUPBY = ("node", "type", "name")


class NodeProfiler(TimerProfiler):
    """Profiling class for estimating the time of individual nodes."""

    __slots__ = ()

    def __init__(
        self,
        target_nodes: Sequence[Node] = (),
        *,
        sources: Sequence[Node] = (),
        sinks: Sequence[Node] = (),
        n_runs: int = 10_000,
    ):
        super().__init__(target_nodes, sources, sinks, n_runs)
        self._allowed_groupby = _ALLOWED_GROUPBY
        self._primary_col = "time"

    @classmethod
    def estimate_node(cls, node: Node, n_runs: int = 10_000):
        for input in node.inputs.iter_all():
            input.touch()
        return timeit(stmt=node.function, number=n_runs)

    def estimate_target_nodes(self) -> NodeProfiler:
        """
        Estimate all nodes in `self.target_nodes`.

        Return the current `NodeProfiler` instance.
        """
        self._estimations_table = DataFrame({
            "node": (str(node) for node in self._target_nodes),
            "type": (type(node).__name__ for node in self._target_nodes),
            "name": (node.name for node in self._target_nodes),
            "time": (self.estimate_node(node, self._n_runs)
                     for node in self._target_nodes)
        })
        return self


    def make_report(
        self,
        *,
        group_by: str | Sequence[str] | None = "type",
        aggregations: Sequence[str] | None = None,
        sort_by: str | None = None  ,
        average_by_runs: bool = True,
    ) -> DataFrame:
        report = super().make_report(group_by, aggregations, sort_by)
        if average_by_runs:
            return self._compute_average(report)
        return report

    def _print_total_time(self):
        total = self._total_estimations_time()
        print(f"total estimations time / n_runs: {total / self._n_runs:.9f} sec.")
        print(f"total estimations time: {total:.6f} sec.")

    def print_report(
        self,
        *,
        rows: int | None = 40,
        group_by: str | Sequence[str] | None = "type",
        aggregations: Sequence[str] | None = None,
        sort_by: str | None = None,
    ) -> DataFrame:
        report = self.make_report(
            group_by=group_by,
            aggregations=aggregations,
            sort_by=sort_by
        )
        print(
            f"\nNode Profiling {hex(id(self))}, "
            f"n_runs for each node: {self._n_runs}\n"
            f"sort by: `{sort_by or 'default sorting'}`, "
            f"group by: `{group_by or 'no grouping'}`"
        )
        super()._print_table(report, rows)
        self._print_total_time()
        return report
