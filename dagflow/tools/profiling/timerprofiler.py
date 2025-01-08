from __future__ import annotations

from typing import TYPE_CHECKING

import numpy
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
_AGG_ALIASES: dict[str, str | Callable] = {
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

_DEFAULT_AGG_FUNCS = ("count", "single", "sum", "%_of_total")


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
        self._default_agg_funcs = _DEFAULT_AGG_FUNCS
        self._column_aliases = _COLUMN_ALIASES.copy()
        self._agg_aliases = _AGG_ALIASES.copy()
        self._n_runs = n_runs
        self.register_agg_func(
            func=self._t_presentage,
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

    def _t_presentage(self, _s: Series) -> Series:
        """User-defined aggregate function to calculate the percentage of group
        given as `pandas.Series`."""
        total = self._total_estimations_time()
        return Series({"%_of_total": numpy.sum(_s) * 100 / total})

    def _total_estimations_time(self):
        return self._estimations_table["time"].sum()

    def _compute_average(self, df: DataFrame) -> DataFrame:
        """Normalize `time` by `self.n_runs`"""
        for c in df.columns:
            if c.startswith("t_") or c == "time":
                df[c] /= self._n_runs
        return df
