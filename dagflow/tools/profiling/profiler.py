from __future__ import annotations

from typing import TYPE_CHECKING, overload
from collections import deque
from abc import ABCMeta, abstractmethod

from pandas import DataFrame, Index
from tabulate import tabulate

if TYPE_CHECKING:
    from dagflow.nodes import FunctionNode
    from collections.abc import Generator, Sequence, Iterable


# prefix `t_` - time notation
_COLUMN_ALIASES: dict[str, str] = {
    "mean": "t_single",
    "single": "t_single",
    "t_mean": "t_single",
    "median": "t_median",
    "sum": "t_sum",
    "std": "t_std",
    "t_count": "count",
    "min": "t_min",
    "var": "t_var",
    "t_percentage": "%_of_total",
    "percentage": "%_of_total",
}
_AGG_ALIASES: dict[str, str] = {
    "single": "mean",
    "t_single": "mean",
    "t_mean": "mean",
    "t_median": "median",
    "t_sum": "sum",
    "t_std": "std",
    "t_count": "count",
    "t_min": "min",
    "t_var": "var",
    "t_percentage": "%_of_total",
    "percentage": "%_of_total"
}

_ALLOWED_AGG_FUNCS = ("count", "mean", "median", "std", "min", "max", "sum",
                      "var", "%_of_total")
_DEFAULT_AGG_FUNCS = ("count", "single", "sum", "%_of_total")


class Profiler(metaclass=ABCMeta):
    """Base Profiler Class."""
    __slots__ = (
        "_target_nodes",
        "_sources",
        "_sinks",
        "_n_runs",
        "_estimations_table",
        "_ALLOWED_GROUPBY",
        "_ALLOWED_AGG_FUNCS",
        "_DEFAULT_AGG_FUNCS"
    )
    _target_nodes: Sequence[FunctionNode]
    _sources: Sequence[FunctionNode]
    _sinks: Sequence[FunctionNode]
    _n_runs: int
    _estimations_table: DataFrame
    _ALLOWED_GROUPBY: tuple[list[str] | str, ...]
    _ALLOWED_AGG_FUNCS: tuple[str, ...]
    _DEFAULT_AGG_FUNCS: tuple[str, ...]

    def __init__(self,
                 target_nodes: Sequence[FunctionNode]=[],
                 sources: Sequence[FunctionNode]=[],
                 sinks: Sequence[FunctionNode]=[],
                 n_runs: int=100):
        self._ALLOWED_AGG_FUNCS = _ALLOWED_AGG_FUNCS
        self._DEFAULT_AGG_FUNCS = _DEFAULT_AGG_FUNCS
        self._sources = sources
        self._sinks = sinks
        self._n_runs = n_runs
        if target_nodes:
            self._target_nodes = target_nodes
        elif sources and sinks:
            self._target_nodes = list(self._gather_related_nodes())
        else:
            raise ValueError("You shoud provide profiler with `target_nodes` "
                             "or provide `sources` and `sinks` arguments"
                             "to automatically find the target nodes")

    def __child_nodes_gen(self, node: FunctionNode) \
                            -> Generator[FunctionNode, None, None]:
        for output in node.outputs.iter_all():
            for child_input in output.child_inputs:
                yield child_input.node

    def __parent_nodes_gen(self, node: FunctionNode) \
                            -> Generator[FunctionNode, None, None]:
        for input in node.inputs.iter_all():
            yield input.parent_node

    def __check_reachable(self, nodes_gathered):
        if any(s not in nodes_gathered for s in self._sinks):
            raise ValueError("Some of the `sinks` nodes are unreachable "
                             "(no paths from sources)")

    def _gather_related_nodes(self) -> set[FunctionNode]:
        """Find all nodes that lie on all possible paths
        between `self._sources` and `self._sinks`
        """
        nodes_stack = deque()
        iters_stack = deque()
        related_nodes = set(self._sources)
        for start_node in self._sources:
            current_iterator = self.__child_nodes_gen(start_node)
            while True:
                try:
                    node = next(current_iterator)
                    nodes_stack.append(node)
                    iters_stack.append(current_iterator)
                    if node in self._sinks:
                        related_nodes.update(nodes_stack)
                        nodes_stack.pop()
                        current_iterator = iters_stack.pop()
                    else:
                        current_iterator = self.__child_nodes_gen(node)
                except StopIteration:
                    if len(nodes_stack) == 0:
                        break
                    nodes_stack.pop()
                    current_iterator = iters_stack.pop()
        self.__check_reachable(related_nodes)
        return related_nodes

    def _reveal_source_sink(self):
        """Find sources and sinks for self._target_nodes"""
        sources = []
        sinks = []
        for node in self._target_nodes:
            have_parents = any(n in self._target_nodes
                            for n in self.__parent_nodes_gen(node))
            have_childs = any(n in self._target_nodes
                           for n in self.__child_nodes_gen(node))
            if have_parents and have_childs:
                continue
            elif have_parents:
                sinks.append(node)
            else:
                sources.append(node)
        self._sources = sources
        self._sinks = sinks

    def _cols_from_aliases(self, aliases: Iterable[str]) -> list[str]:
        """Return the column names if aliases exists,
        otherwise return the same strings.
        """
        return [_COLUMN_ALIASES.get(al, al) for al in aliases]
        
    def _col_from_alias(self, alias: str | None) -> str | None:
        """Return the column name if an alias exists,
        otherwise return the same string.
        """
        return _COLUMN_ALIASES.get(alias, alias)
    
    def _aggs_from_aliases(self, aliases: Iterable[str]) -> list[str]:
        """Return aggregate function names if aliases exists,
        otherwise return the same strings.
        """
        return [_AGG_ALIASES.get(al, al) for al in aliases]
    
    def _agg_from_alias(self, alias: str | None) -> str | None:
        """Return aggregate function name if an alias exists,
        otherwise return the same string.
        """
        return _AGG_ALIASES.get(alias, alias)

    def _pd_funcs_agg_df(self, grouped_df, grouped_by, agg_funcs) -> DataFrame:
        """Apply standard Pandas aggregate
        functions (`"min"`, `"max"`, etc.)
        to the given grouped `DataFrame`.
        """
        df = grouped_df.agg({'time': agg_funcs})
        # grouped_by can be ["col1", "col2", ...] or "col"
        if isinstance(grouped_by, list):
            new_columns = grouped_by.copy()
        else:
            new_columns = [grouped_by]
        # get rid of multiindex
        new_columns += self._cols_from_aliases(agg_funcs)
        df.columns = Index(new_columns)
        return df

    def __get_index_and_pop(self, array: list, value):
        """Return index of the `value` in given `array` and pop it.
        Return `-1` if index not exists. \n
        Helper function for `_aggregate_df`
        """
        try:
            idx = array.index(value)
            array.pop(idx)
            return idx
        except ValueError:
            return -1

    def _aggregate_df(self, grouped_df, grouped_by, agg_funcs) -> DataFrame:
        """Apply the aggregate Pandas functions
        and calculate the percentage `"%_of_total"` separately
        if it is specified as an aggregate function
        """
        tmp_aggs = self._aggs_from_aliases(agg_funcs)
        p_index = self.__get_index_and_pop(tmp_aggs, '%_of_total')
        if p_index != -1 and 'sum' not in tmp_aggs:
            tmp_aggs = tmp_aggs + ['sum']
        df = self._pd_funcs_agg_df(grouped_df, grouped_by, tmp_aggs)
        if p_index != -1:
            total_time = df['t_sum'].sum()
            df.insert(len(df.columns) - len(agg_funcs) + p_index,
                      '%_of_total', df['t_sum'] * 100 / total_time)
        if p_index != -1 and 'sum' not in agg_funcs:
            df.drop('t_sum', inplace=True, axis=1)
        df.columns = Index(self._cols_from_aliases(df.columns))
        return df

    def _check_report_consistency(self, group_by, agg_funcs):
        """Check if it is possible to create a report table
        """
        if not hasattr(self, "_estimations_table"):
            raise AttributeError("No estimations found!\n"
                                 "Note: first esimate your nodes "
                                 "with methods like `estimate_*`")
        if group_by != None and (hasattr(self, "_ALLOWED_GROUPBY") and
                                 group_by not in self._ALLOWED_GROUPBY):
            raise ValueError(f"Invalid `group_by` name \"{group_by}\"."
                             f"You must use one of these: {self._ALLOWED_GROUPBY}")
        if any(self._agg_from_alias(a) not in self._ALLOWED_AGG_FUNCS
               for a in agg_funcs):
            raise ValueError("Invalid aggregate function"
                             "You should use one of these:"
                             f"{self._ALLOWED_AGG_FUNCS}")

    @abstractmethod
    def make_report(self,
                    group_by: str | tuple[str] | None,
                    agg_funcs: Sequence[str] | None,
                    sort_by: str | None) -> DataFrame:
        """Make a report table. \n
        Note: Since the report table is just a `Pandas.DataFrame`,
        you can call Pandas methods like `.to_csv()` or `to_excel()`
        """
        if agg_funcs is None or agg_funcs == []:
            agg_funcs = self._DEFAULT_AGG_FUNCS
        self._check_report_consistency(group_by, agg_funcs)
        sort_by = self._col_from_alias(sort_by)
        report = self._estimations_table.copy()
        if group_by is None:
            report.sort_values(sort_by or 'time', ascending=False,
                               ignore_index=True, inplace=True)
        else:
            grouped = report.groupby(group_by, as_index=False)
            report = self._aggregate_df(grouped, group_by, agg_funcs)
            if sort_by is None:
                sort_by = self._col_from_alias( agg_funcs[0] )
            report.sort_values(sort_by, ascending=False,
                               ignore_index=True, inplace=True)
        return report

    def _normalize(self, df: DataFrame) -> DataFrame:
        """Normalize time by `self._n_runs`"""
        for c in df.columns:
            if c.startswith('t_') or c == 'time':
                df[c] /= self._n_runs
        return df

    def _print_table(self, df: DataFrame, rows):
        print(tabulate(df.head(rows), headers='keys', tablefmt='psql'))

    @abstractmethod
    def print_report(self,
                     rows: int | None,
                     group_by: str | None,
                     agg_funcs: Sequence[str] | None,
                     sort_by) -> DataFrame:
        """Make report and print it. \n
        Return `Pandas.DataPrame` as report
        ( See: `self.make_report()` )
        """
        report = self.make_report(group_by, agg_funcs, sort_by)
        self._print_table(report, rows)
        raise NotImplementedError(
            "You must override `print_report` in subclass"
        )

