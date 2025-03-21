from __future__ import annotations
from collections.abc import Sequence
from typing import Literal
from timeit import timeit, repeat

from pandas import DataFrame

from dagflow.core.node import Node
from dagflow.lib.common import Array

from .timer_profiler import TimerProfiler


class FitSimulationProfiler(TimerProfiler):
    """Profiler class for simulating model fit process.

    This class inherits from TimerProfiler and uses source nodes
    as tweakable params to imitate model fit process.
    """

    __slots__ = ("_fit_step", "_mode")

    def __init__(
        self,
        mode: Literal["element-wise", "simultaneous"] = "element-wise",
        tweakable_params: Sequence[Array] = (),
        endpoints: Sequence[Node] = (),
        # *,
        # sources: Sequence[Node] = (),
        # sinks: Sequence[Node] = (),
        n_runs: int = 10_000,
    ):
        self.mode = mode
        super().__init__(sources=tweakable_params, sinks=endpoints, n_runs=n_runs)
        # self._allowed_groupby = _ALLOWED_GROUPBY
        self._primary_col = "time"
        self._default_aggregations = ("count", "single", "sum")

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        if value == "element-wise":
            self._fit_step = self._separate_step
        elif value == "simultaneous":
            self._fit_step = self._together_step
        else:
            raise ValueError(f"Unknown mode: {value}")
        self._mode = value

    @property
    def params(self):
        return self._sources

    @property
    def endpoints(self):
        return self._sinks

    def _together_step(self):
        for param in self._sources:
            param.taint()
        self.__call_endpoints()
        # probably backprop should be here

    def _separate_step(self):
        for param in self._sources:
            # simulate finding derivative by 4 points
            for _ in range(4):
                param.taint()
                self.__call_endpoints()
            # simulate reverting to the initial state
            param.taint()

        # make a step for all params
        for param in self._sources:
            param.taint()
        self.__call_endpoints()

    def __call_endpoints(self):
        # TODO: probably there is should be a better name for this function
        for endpoint in self._sinks:
            # TODO: remove this assertion. It is for development only
            assert endpoint.tainted
            endpoint.touch()

    def _touch_model_nodes(self):
        for param_node in self._target_nodes:
            param_node.touch()

    def estimate_fit(self) -> FitSimulationProfiler:
        self._touch_model_nodes()
        results = repeat(self._fit_step, setup='pass', repeat=self.n_runs, number=1)
        source_short_names, sink_short_names = self._shorten_sources_sinks()
        self._estimations_table = DataFrame({
            "tweakable params": source_short_names,
            "endpoints": sink_short_names,
            "eval mode": self._mode,
            "time": results
        })
        return self

    def make_report(
        self,
        group_by: str | Sequence[str] | None = ("tweakable params", "endpoints", "eval mode"),
        aggregations: Sequence[str] | None = None,
        sort_by: str | None = None,
    ) -> DataFrame:
        return super().make_report(group_by, aggregations, sort_by)

    def print_report(
        self,
        rows: int | None = 40,
        group_by: str | Sequence[str] | None = ("tweakable params", "endpoints", "eval mode"),
        aggregations: Sequence[str] | None = None,
        sort_by: str | None = None,
    ) -> DataFrame:
        report = self.make_report(
            group_by=group_by,
            aggregations=aggregations,
            sort_by=sort_by
        )
        print(
            f"\nFit simulation Profiling {hex(id(self))}, "
            f"fit steps (n_runs): {self._n_runs}, "
            f"nodes in subgraph: {len(self._target_nodes)}\n"
            f"sort by: `{sort_by or 'default sorting'}`, "
            f"group by: `{group_by or 'no grouping'}`"
        )
        super()._print_table(report, rows)
        return report
