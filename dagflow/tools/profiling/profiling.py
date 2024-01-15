from __future__ import annotations

import collections
from collections.abc import Generator, Sequence
from abc import ABCMeta, abstractmethod

import pandas as pd
import tabulate

from dagflow.nodes import FunctionNode
from dagflow.iter import IsIterable


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
_AGG_ALIASES: dict[str: str] = {
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

# abc - Abstract Base Class
class Profiling(metaclass=ABCMeta):
    _target_nodes: Sequence[FunctionNode]
    _source: Sequence[FunctionNode]
    _sink: Sequence[FunctionNode]
    _n_runs: int
    _estimations_table: pd.DataFrame
    _ALLOWED_GROUPBY: tuple[str]
    _ALLOWED_AGG_FUNCS: tuple[str] = ("count", "mean", "median",
                                      "std",   "min",  "max",
                                      "sum",   "var",  "%_of_total")
    _DEFAULT_AGG_FUNCS: tuple[str] = ("count", "single", "sum", "%_of_total")

    def __init__(self,
                 target_nodes: Sequence[FunctionNode]=[],
                 source: Sequence[FunctionNode]=[], 
                 sink: Sequence[FunctionNode]=[], 
                 n_runs: int=100):
        self._source = source
        self._sink = sink
        self._n_runs = n_runs
        if target_nodes:
            self._target_nodes = target_nodes
        elif source and sink:
            self._target_nodes = list(self._gather_related_nodes())
        else:
            raise ValueError("You shoud provide profiler with `target_nodes` "
                             "or use `source` and `sink` to find "
                             "target nodes automaticly")

    def __child_nodes_gen(self, node: FunctionNode) -> Generator[FunctionNode, None, None]:
        for output in node.outputs.iter_all():
            for child_input in output.child_inputs:
                yield child_input.node

    def __parent_nodes_gen(self, node: FunctionNode) -> Generator[FunctionNode, None, None]:
        for input in node.inputs.iter_all():
            yield input.parent_node

    def __check_reachable(self, nodes_gathered):
        if not all(s in nodes_gathered for s in self._sink):
            raise ValueError("Some of the `sink` nodes are unreachable "
                             "(no paths from source)")

    def _gather_related_nodes(self) -> set[FunctionNode]:
        nodes_stack = collections.deque()
        iters_stack = collections.deque()
        related_nodes = set(self._source)
        for start_node in self._source:
            current_iterator = self.__child_nodes_gen(start_node)
            while True:
                try:
                    node = next(current_iterator)
                    nodes_stack.append(node)
                    iters_stack.append(current_iterator)
                    if node in self._sink:
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
        source = []
        sink = []
        for node in self._target_nodes:
            have_parents = any(n in self._target_nodes 
                            for n in self.__parent_nodes_gen(node))
            have_childs = any(n in self._target_nodes 
                           for n in self.__child_nodes_gen(node))
            if have_parents and have_childs:
                continue
            elif have_parents:
                sink.append(node)
            else:
                source.append(node)
        self._source = source
        self._sink = sink

    def __anything_from_alias(self, alias_es, alias_table: dict):
        def get_orig(val):
            try:
                return alias_table.get(val, val)
            except TypeError:   # protect from unhashable types like `list`
                return val 
        if IsIterable(alias_es):
            return [get_orig(a) for a in alias_es]
        return get_orig(alias_es)

    def _cols_from_alias(self, alias_es: list | str | None):
        """Returns the column name(s) if an alias(es) exists, 
        otherwise returns the same object - `alias_es`"""
        return self.__anything_from_alias(alias_es, _COLUMN_ALIASES)
    
    def _aggs_from_alias(self, alias_es: str | None) -> str | None:
        """Returns aggregate function name(s) if an alias(es) exists, 
        otherwise returns the same object - `alias_es`"""
        return self.__anything_from_alias(alias_es, _AGG_ALIASES)

    def _pd_funcs_agg_df(self, grouped_df, grouped_by, agg_funcs) -> pd.DataFrame:
        df = grouped_df.agg({'time': agg_funcs})
        # grouped_by can be ["col1", "col2", ...] or "col"
        new_columns = grouped_by.copy() if type(grouped_by)==list else [grouped_by]
        # get rid of multiindex
        new_columns += [self._cols_from_alias(c) for c in agg_funcs]
        df.columns = new_columns
        return df
    
    def __get_index_and_pop(self, array: list, value):
        try:
            idx = array.index(value)
            array.pop(idx)
            return idx
        except ValueError:
            return -1
    
    def _aggregate_df(self, grouped_df, grouped_by, agg_funcs) -> pd.DataFrame:
        tmp_aggs = self._aggs_from_alias(agg_funcs)
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
        df.columns = self._cols_from_alias(df.columns)
        return df
    
    def _check_report_capability(self, group_by, agg_funcs):
        if not hasattr(self, "_estimations_table"):
            raise AttributeError("No estimations found!\n"
                                 "Note: first esimate your nodes "
                                 "with methods like `estimate_*`")
        if group_by != None and (hasattr(self, "_ALLOWED_GROUPBY") and 
                                 group_by not in self._ALLOWED_GROUPBY):
            raise ValueError(f"Invalid `group_by` name \"{group_by}\"."
                             f"You must use one of these: {self._ALLOWED_GROUPBY}")
        if not all(self._aggs_from_alias(a) in self._ALLOWED_AGG_FUNCS
                   for a in agg_funcs):
            raise ValueError("Invalid aggregate function"
                             "You should use one of these:"
                             f"{self._ALLOWED_AGG_FUNCS}")
            
    @abstractmethod
    def make_report(self, group_by, agg_funcs, sort_by) -> pd.DataFrame:
        if agg_funcs == None or agg_funcs == []:
            agg_funcs = self._DEFAULT_AGG_FUNCS
        self._check_report_capability(group_by, agg_funcs)
        sort_by = self._cols_from_alias(sort_by)
        report = self._estimations_table.copy()
        if group_by == None:
            report.sort_values(sort_by or 'time', ascending=False,
                               ignore_index=True, inplace=True)
        else:
            grouped = report.groupby(group_by, as_index=False)
            report = self._aggregate_df(grouped, group_by, agg_funcs)
            if sort_by == None:
                sort_by = self._cols_from_alias(agg_funcs[0])
            report.sort_values(sort_by, ascending=False,
                               ignore_index=True, inplace=True)
        return report
    
    def _normilize(self, df: pd.DataFrame) -> pd.DataFrame:
        for c in df.columns:
            if c.startswith('t_') or c == 'time':
                df[c] /= self._n_runs
        return df
    
    def _print_table(self, df: pd.DataFrame, rows):
        print(tabulate.tabulate(df.head(rows), 
                                headers='keys', 
                                tablefmt='psql'))
        
    def _print_total_time(self):
        total = self._estimations_table['time'].sum()
        print("total estimations time"
              " / n_runs: %.9f sec." % (total / self._n_runs))
        print("total estimations time: %.6f sec." % total)
    
    @abstractmethod
    def print_report(self, rows, *args, **kwargs):
        report = self.make_report(*args, **kwargs)
        self._print_table(report, rows)
        raise NotImplementedError

