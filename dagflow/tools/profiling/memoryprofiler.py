from __future__ import annotations

from typing import TYPE_CHECKING

from pandas import DataFrame

if TYPE_CHECKING:
    from collections.abc import Sequence, Callable
    from dagflow.node import Node
    from dagflow.input import Input
    from dagflow.output import Output

from .profiler import Profiler

# prefix `s_` - size notation
# columnt aliases for aggrigate functions
_COLUMN_ALIASES: dict[str | Callable, str] = {
    "mean": "size_single",
    "single": "size_single",
    "size_mean": "size_single",
    "median": "size_median",
    "sum": "size_sum",
    "std": "size_std",
    "size_count": "count",
    "min": "size_min",
    "max": "size_max",
    "var": "size_var",
}
_AGG_ALIASES: dict[str, str | Callable] = {
    "single": "mean",
    "size_single": "mean",
    "size_mean": "mean",
    "size_median": "median",
    "size_sum": "sum",
    "size_std": "std",
    "size_count": "count",
    "size_min": "min",
    "size_max": "max",
    "size_var": "var",
}

_DEFAULT_AGG_FUNCS = ("count", "sum")


class MemoryProfiler(Profiler):
    def __init__(
        self,
        target_nodes: Sequence[Node]=(),
        *,
        sources: Sequence[Node]=(),
        sinks: Sequence[Node]=()
    ):
        self._default_agg_funcs = _DEFAULT_AGG_FUNCS
        self._column_aliases = _COLUMN_ALIASES.copy()
        self._agg_aliases = _AGG_ALIASES.copy()

        self._primary_col = "size"

        super().__init__(target_nodes, sources, sinks, n_runs=1)

    def _touch_nodes(self):
        for node in self._target_nodes:
            node.eval()

    @classmethod
    def estimate_node(cls, node) -> dict[Input | Output, int]:
        """Return `dict` of sizes for each Input/Output of given `node`"""
        estimations = {}
        inp: Input
        out: Output
        for inp in node.inputs.iter_all():
            if inp.has_data and inp.owns_buffer:
                estimations[inp] = inp._own_data.nbytes
            else:
                estimations[inp] = 0
        for out in node.outputs.iter_all():
            if (out.has_data 
                and (out.owns_buffer or out._allocating_input is None)):
                # If there is an _allocating_input, 
                #  the `out.data` refers to the child `Input` data.
                # However if there is no `_allocating_input` 
                #  and owns_buffer=False (and `out.data` is not `None` of course)
                #  then it means there is allocated memory for this Output.
                estimations[out] = out.data_unsafe.nbytes
            else:
                estimations[out] = 0
        return estimations


    def estimate_target_nodes(self, touch=True):
        """Estimates size of all edges of all `self.target_nodes`.
        
        Return current `MemoryProfiler` instance.
        """
        data_sizes = {}
        if touch:
            self._touch_nodes()
        
        records = {col: [] for col in ("node", "type", "edge", "size")}
        for node in self._target_nodes:
            estimations = self.estimate_node(node)
            for edge, size in estimations.items():
                records["node"].append(str(node))
                records["type"].append(type(node).__name__)
                records["edge"].append(str(edge))
                records["size"].append(size)
        self._estimations_table = DataFrame(records)
        return self
    
    @property
    def total_size(self):
        """Return size of all edges of '_target_nodes' in bytes
        """
        if not hasattr(self, '_estimations_table'):
            self.estimate_target_nodes()
        return self._estimations_table["size"].sum()

    def make_report(
        self,
        group_by: str | list[str] | None = "type",
        agg_funcs: Sequence[str] | None = None,
        sort_by: str | None = None
    ) -> DataFrame:
        return super().make_report(group_by, agg_funcs, sort_by)
    
    def print_report(
        self,
        rows: int | None = 40,
        group_by: str | list[str] | None = "type",
        agg_funcs: Sequence[str] | None = None,
        sort_by: str | None = None
    ) -> DataFrame:
        # TODO: add default args
        # TODO: test group_by
        report = self.make_report(group_by, agg_funcs, sort_by)
        print(f"\nMemory Profiling {hex(id(self))}, "
              f"sort by: `{sort_by or 'default sorting'}`, "
              f"group by: `{group_by or 'no grouping'}`")
        self._print_table(report, rows)
        size_bytes = self.total_size
        print(f"TOTAL SIZE:\t{size_bytes} bytes\n"
              f"\t\t{size_bytes / 2 ** 10} Kbytes\n"
              f"\t\t{size_bytes / 2 ** 20} Mbytes")
        return report
