from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections import deque
from collections.abc import Sequence
from typing import TYPE_CHECKING

from pandas import DataFrame, Index
from tabulate import tabulate

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable

    from pandas.api.typing import DataFrameGroupBy

    from dagflow.core.node import Node


class Profiler(metaclass=ABCMeta):
    """Base Profiler Class."""

    __slots__ = (
        "_target_nodes",
        "_sources",
        "_sinks",
        "_estimations_table",
        "_allowed_groupby",
        "_default_aggregations",
        "_primary_col",
        "_aggregate_aliases",
        "_column_aliases",
    )
    _target_nodes: Sequence[Node]
    _sources: Sequence[Node]
    _sinks: Sequence[Node]
    _estimations_table: DataFrame
    _allowed_groupby: tuple[Sequence[str] | str, ...]
    _default_aggregations: tuple[str, ...]
    _primary_col: str
    _column_aliases: dict[str | Callable, str]
    _aggregate_aliases: dict[str, str | Callable]

    def __init__(
        self,
        target_nodes: Sequence[Node] = (),
        sources: Sequence[Node] = (),
        sinks: Sequence[Node] = (),
    ):
        self._sources = sources
        self._sinks = sinks
        if target_nodes:
            self._target_nodes = target_nodes
        elif sources and sinks:
            self._target_nodes = list(self._gather_related_nodes())
        else:
            raise ValueError(
                "You shoud provide profiler with `target_nodes` "
                "or provide `sources` and `sinks` arguments "
                "to automatically find the target nodes"
            )

    def __child_nodes_gen(self, node: Node) -> Generator[Node, None, None]:
        """Access to the child nodes of the given node via the generator."""
        for output in node.outputs.iter_all():
            for child_input in output.child_inputs:
                yield child_input.node

    def __parent_nodes_gen(self, node: Node) -> Generator[Node, None, None]:
        """Access to the parent nodes of the given node via the generator."""
        for input in node.inputs.iter_all():
            yield input.parent_node

    def __check_reachable(self, nodes_gathered):
        for sink in self._sinks:
            if sink not in nodes_gathered:
                raise ValueError(
                    f"One of the `sinks` nodes is unreachable: {sink} " "(no paths from sources)"
                )

    def _gather_related_nodes(self) -> set[Node]:
        """Find all nodes that lie on all possible paths between
        `self._sources` and `self._sinks`

        Modified Depth-first search (DFS) algorithm for multiple sources
        and sinks
        """
        related_nodes = set(self._sources)
        # Deque works well as Stack
        stack = deque()
        visited = set()
        for start_node in self._sources:
            cur_node = start_node
            while True:
                last_in_path = True
                for ch in self.__child_nodes_gen(cur_node):
                    if ch in self._sinks:
                        related_nodes.add(ch)
                    # If `_sinks` contains child node it would be already in `related_nodes`
                    if ch in related_nodes:
                        related_nodes.update(stack)
                        related_nodes.add(cur_node)
                    elif ch not in visited:
                        stack.append(cur_node)
                        cur_node = ch
                        last_in_path = False
                        break
                # No unvisited childs found (`for` loop did not encounter a `break`)
                else:
                    visited.add(cur_node)
                if len(stack) == 0:
                    break
                if last_in_path:
                    cur_node = stack.pop()
        self.__check_reachable(related_nodes)
        return related_nodes

    def _reveal_source_sink(self):
        """Find sources and sinks for self._target_nodes."""
        sources = []
        sinks = []
        for node in self._target_nodes:
            have_parents = any(n in self._target_nodes for n in self.__parent_nodes_gen(node))
            have_childs = any(n in self._target_nodes for n in self.__child_nodes_gen(node))
            if have_parents and have_childs:
                continue
            elif have_parents:
                sinks.append(node)
            else:
                sources.append(node)
        self._sources = sources
        self._sinks = sinks

    def register_aggregate_func(
        self, func: Callable, aliases: Sequence[str], column_name: str
    ) -> None:
        """Add user-defined function for the Profiler.

        The function applied for each group in grouped
        data (i.e. data returned by `pandas.DataFrame.groupby()`)
        separately.
        """
        for al in aliases:
            self._aggregate_aliases[al] = func
            self._column_aliases[al] = column_name
        self._column_aliases[func] = column_name

    def _cols_from_aliases(self, aliases: Iterable[str]) -> list[str]:
        """Return the column names if aliases exists, otherwise return the same
        strings."""
        return [self._column_aliases.get(al, al) for al in aliases]

    def _col_from_alias(self, alias: str) -> str:
        """Return the column name if an alias exists, otherwise return the same
        object."""
        return self._column_aliases.get(alias, alias)

    def _aggregations_from_aliases(self, aliases: Iterable[str]) -> list[str | Callable]:
        """Return aggregate function names if aliases exists, otherwise return
        the same object."""
        return [self._aggregate_aliases.get(al, al) for al in aliases]

    def _aggregate_from_alias(self, alias: str) -> str | Callable:
        """Return aggregate function name if an alias exists, otherwise return
        the same object."""
        return self._aggregate_aliases.get(alias, alias)

    def _aggregate_df(
        self,
        grouped_df: DataFrameGroupBy,
        grouped_by: str | Sequence[str],
        aggregate_names: Sequence[str],
    ) -> DataFrame:
        """Apply pandas built-ins and user-defined aggregate functions (given
        as their aliases) on the `self._primary_col` column of the grouped data
        `grouped_df`
        """
        aggregate_funcs = self._aggregations_from_aliases(aggregate_names)
        df = grouped_df.agg({self._primary_col: aggregate_funcs})
        # grouped_by can be ["col1", "col2", ...] or "col"
        if isinstance(grouped_by, list):
            new_columns = grouped_by.copy()
        else:
            new_columns = [grouped_by]
        # get rid of multiindex
        new_columns += self._cols_from_aliases(aggregate_names)
        df.columns = Index(new_columns)
        return df

    def __possible_aggregations(self):
        """Return set of all possible values for `aggregations`
        argument of `make_report` and `print_report` methods.

        Helper method for `_check_report_consistency`.
        Called on exception.
        """
        values = set(self._aggregate_aliases.keys())
        for agg_name in self._aggregate_aliases.values():
            if isinstance(agg_name, str):
                values.add(agg_name)
        return values

    def _check_report_consistency(self, group_by, aggregate_funcs):
        """Check if it is possible to create a report table."""
        if not hasattr(self, "_estimations_table"):
            raise AttributeError(
                "No estimations found!\n"
                "Note: first esimate your nodes "
                "with methods like `estimate_*`"
            )
        if group_by is not None and (
            hasattr(self, "_allowed_groupby") and group_by not in self._allowed_groupby
        ):
            raise ValueError(
                f'Invalid `group_by` name "{group_by}".\n'
                f"You must use one of these: {', '.join(map(str, self._allowed_groupby))}"
            )
        for a in self._aggregations_from_aliases(aggregate_funcs):
            if a not in self._aggregate_aliases.values():
                raise ValueError(
                    f"Invalid aggregate function `{a}`. "
                    "You should use one of these: "
                    f"{self.__possible_aggregations()}"
                )

    @abstractmethod
    def make_report(
        self,
        group_by: str | Sequence[str] | None,
        aggregations: Sequence[str] | None,
        sort_by: str | None,
    ) -> DataFrame:
        """Make a report table.

        Note: Since the report table is just a `Pandas.DataFrame`,
        you can call Pandas methods like `.to_csv()` or `.to_excel()`
        to export your data in appropriate format.
        """
        aggregate_names = aggregations or self._default_aggregations
        self._check_report_consistency(group_by, aggregate_names)
        sort_by = self._col_from_alias(sort_by) if sort_by else None
        report = self._estimations_table.copy()
        if group_by is None:
            sort_by = sort_by or self._primary_col
        else:
            if isinstance(group_by, Sequence) and not isinstance(group_by, str):
                group_by = list(group_by)  # pandas accepts only lists
            grouped = report.groupby(group_by, as_index=False)
            report = self._aggregate_df(grouped, group_by, aggregate_names)
            if sort_by is None:
                sort_by = self._col_from_alias(aggregate_names[0])
        report.sort_values(sort_by, ascending=False, ignore_index=True, inplace=True)
        return report

    def _print_table(self, df: DataFrame, rows, *, float_fmt="g", int_fmt=","):
        print(
            tabulate(
                tabular_data=df.head(rows),  # type: ignore
                headers="keys",
                tablefmt="psql",
                floatfmt=float_fmt,
                intfmt=int_fmt,
            )
        )
        if len(df) > rows:
            print(f" --- showing first {rows} rows --- \n")

    @abstractmethod
    def print_report(
        self,
        rows: int | None,
        group_by: str | Sequence[str] | None,
        aggregations: Sequence[str] | None,
        sort_by: str | None,
    ) -> DataFrame:
        """Make report and print it.

        Return `Pandas.DataPrame` as report
        ( See: `make_report()` method )
        """
        report = self.make_report(group_by, aggregations, sort_by)
        self._print_table(report, rows)
        raise NotImplementedError("You must override `print_report` in subclass")
