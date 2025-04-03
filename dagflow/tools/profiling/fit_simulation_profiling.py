from __future__ import annotations
from collections.abc import Sequence
from typing import Literal
from timeit import repeat

from pandas import DataFrame

from dagflow.core.node import Node

from .timer_profiler import TimerProfiler


_ALLOWED_GROUPBY = (("parameters", "endpoints", "eval mode"),)


class FitSimulationProfiler(TimerProfiler):
    """Profiler class for simulating model fit process.

    This class inherits from TimerProfiler and uses source nodes
    as tweakable parameters to imitate model fit process.
    """

    __slots__ = ("_fit_step", "_mode", "_n_points")

    def __init__(
        self,
        mode: Literal["parameter-wise", "simultaneous"] = "parameter-wise",
        *,
        parameters: Sequence[Node] = (),
        endpoints: Sequence[Node] = (),
        n_runs: int = 10_000,
        param_mode_n_points: int = 4,
    ):
        super().__init__(sources=parameters, sinks=endpoints, n_runs=n_runs)
        if mode == "parameter-wise":
            self._fit_step = self._separate_step
        elif mode == "simultaneous":
            self._fit_step = self._together_step
        else:
            raise ValueError(f"Unknown mode: {mode}")
        self._mode = mode
        self._allowed_groupby = _ALLOWED_GROUPBY
        self._primary_col = "time"
        self._default_aggregations = ("count", "single", "sum")
        self._n_points = param_mode_n_points

    @property
    def mode(self):
        return self._mode

    @property
    def _parameters(self):
        """Alias for `self._sources`"""
        return self._sources

    @property
    def _endpoints(self):
        """Alias for `self._sinks`"""
        return self._sinks

    def _together_step(self):
        for parameter in self._sources:
            parameter.taint()
        self.__call_endpoints()

    def _separate_step(self):
        for parameter in self._sources:
            # simulate finding derivative by N points
            for _ in range(self._n_points):
                parameter.taint()
                self.__call_endpoints()
            # simulate reverting to the initial state
            parameter.taint()

        # make a step for all params
        for parameter in self._sources:
            parameter.taint()
        self.__call_endpoints()

    def __call_endpoints(self):
        for endpoint in self._sinks:
            endpoint.touch()

    def _touch_model_nodes(self):
        for node in self._target_nodes:
            node.touch()

    def estimate_fit(self) -> FitSimulationProfiler:
        self._touch_model_nodes()
        results = repeat(self._fit_step, setup="pass", repeat=self.n_runs, number=1)
        source_short_names, sink_short_names = self._shorten_sources_sinks()
        self._estimations_table = DataFrame({
                "parameters": source_short_names,
                "endpoints": sink_short_names,
                "eval mode": self._mode,
                "time": results,
        })
        return self

    def make_report(
        self,
        group_by: str | Sequence[str] | None = ("parameters", "endpoints", "eval mode"),
        aggregations: Sequence[str] | None = None,
        sort_by: str | None = None,
    ) -> DataFrame:
        return super().make_report(group_by, aggregations, sort_by)

    def print_report(
        self,
        rows: int | None = 40,
        group_by: str | Sequence[str] | None = ("parameters", "endpoints", "eval mode"),
        aggregations: Sequence[str] | None = None,
        sort_by: str | None = None,
    ) -> DataFrame:
        report = self.make_report(
            group_by=group_by, aggregations=aggregations, sort_by=sort_by
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
