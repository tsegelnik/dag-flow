from __future__ import annotations

from textwrap import shorten
from typing import TYPE_CHECKING

from numpy import sum as npsum
from pandas import DataFrame, Series

from dagflow.core.node import Node

from .profiler import Profiler

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from dagflow.core.node import Node


# prefix `t_` - time notation
# columnt aliases for aggrigate functions
_COLUMN_ALIASES: dict[str | Callable, str] = {
    "mean": "t_single",
    "single": "t_single",
    "t_mean": "t_single",
    "median": "t_median",
    "sum": "t_sum",
    "std": "t_std",
    "t_count": "count",
    "min": "t_min",
    "max": "t_max",
    "var": "t_var",
}
_AGGREGATE_ALIASES: dict[str, str | Callable] = {
    "single": "mean",
    "t_single": "mean",
    "t_mean": "mean",
    "t_median": "median",
    "t_sum": "sum",
    "t_std": "std",
    "t_count": "count",
    "t_min": "min",
    "t_max": "max",
    "t_var": "var",
}

_DEFAULT_AGGREGATIONS = ("count", "single", "sum", "%_of_total")

SOURCE_COL_WIDTH = 32
SINK_COL_WIDTH = 32


class TimerProfiler(Profiler):
    """Base class for time-related profiling.

    It is not designed to be used directly,
    you should consider `NodeProfiler` or `FrameworkProfiler`.
    """

    __slots__ = ("_n_runs",)
    _n_runs: int

    def __init__(
        self,
        target_nodes: Sequence[Node] = (),
        sources: Sequence[Node] = (),
        sinks: Sequence[Node] = (),
        n_runs: int = 100,
    ):
        self._default_aggregations = _DEFAULT_AGGREGATIONS
        self._column_aliases = _COLUMN_ALIASES.copy()
        self._aggregate_aliases = _AGGREGATE_ALIASES.copy()
        self._n_runs = n_runs
        self.register_aggregate_func(
            func=self._t_percentage,
            aliases=["%_of_total", "percentage", "t_percentage"],
            column_name="%_of_total",
        )
        super().__init__(target_nodes, sources, sinks)

    @property
    def n_runs(self) -> int:
        return self._n_runs

    @n_runs.setter
    def n_runs(self, value):
        self._n_runs = value

    def _t_percentage(self, _s: Series) -> Series:
        """User-defined aggregate function to calculate the percentage of group
        given as `pandas.Series`."""
        total = self._total_estimations_time()
        if total == 0:
            raise ZeroDivisionError('The total calculated "time" is zero!')
        return Series({"%_of_total": npsum(_s) * 100 / total})

    def _total_estimations_time(self):
        return self._estimations_table["time"].sum()

    def _compute_average(self, df: DataFrame) -> DataFrame:
        """Normalize `time` by `self.n_runs`"""
        for c in df.columns:
            if c.startswith("t_") or c == "time":
                df[c] /= self._n_runs
        return df

    def _shorten_names(self, nodes, max_length) -> str:
        """Get a string representation of names of the `nodes`,
        truncated to not exceed `max_length`.

        Note: This implementation is generally faster than directly applying
        `shorten(str([n.name for n in nodes]), max_length)`.
        """
        names = []
        names_sum_length = 0
        for node in nodes:
            if names_sum_length > max_length:
                break
            names.append(node.name)
            names_sum_length += len(node.name)
        return shorten(", ".join(names), max_length)

    def _shorten_sources_sinks(self) -> tuple[str, str]:
        """Get a short string representation of `sources` and `sinks` names
        with truncation by `SOURCE_COL_WIDTH`, `SINK_COL_WIDTH` lengths.

        Return pair (short names of __sources__, short names of __sinks__)
        """
        return (
            self._shorten_names(self._sources, SOURCE_COL_WIDTH),
            self._shorten_names(self._sinks, SINK_COL_WIDTH),
        )
