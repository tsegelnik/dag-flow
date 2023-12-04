from __future__ import annotations

from timeit import timeit, repeat
import collections
from collections.abc import Generator, Sequence
from abc import ABCMeta, abstractmethod
from textwrap import shorten
import types

import pandas as pd
import tabulate

from dagflow.nodes import FunctionNode

# TODO: split this file by profiling classes


SOURCE_COL_WIDTH = 32
SINK_COL_WIDTH = 32

# abc - Abstract Base Class
class Profiling(metaclass=ABCMeta):
    _target_nodes: Sequence[FunctionNode]
    _source: Sequence[FunctionNode]
    _sink: Sequence[FunctionNode]
    _n_runs: int    
    _estimations_table: pd.DataFrame
    _ALLOWED_GROUPBY: tuple[str]
                                    # 'single' == 'mean' 
    _ALLOWED_AGG_FUNCS: tuple[str] = ("count", "single", "median",
                                      "std", "min", "max",
                                      "sum", "average", "var", "percentage")
    _DEFAULT_AGG_FUNCS: tuple[str] = ("count", "single", "sum", "percentage")

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
            elif have_childs:
                source.append(node)
            else:
                # TODO: remove this exception
                # raise ValueError(f"Node `{node}` unreachable "
                #                  "(has no connections to other given nodes)")
                source.append(node)
        self._source = source
        self._sink = sink

    def _pd_funcs_agg_df(self, grouped_df, grouped_by, agg_funcs) -> pd.DataFrame:
        df = grouped_df.agg({'time': agg_funcs})
        # grouped_by can be ["col1", "col2", ...] or "col"
        new_columns = grouped_by.copy() if type(grouped_by)==list else [grouped_by]
        # get rid of multiindex and add prefix `t_` - time notation
        new_columns += ['t_' + c if c != 'count' else 'count' for c in agg_funcs]
        df.columns = new_columns
        return df
    
    def _get_index_and_pop(self, array: list, value):
        try:
            idx = array.index(value)
            array.pop(idx)
            return idx
        except ValueError:
            return -1
        
    def _normilize(self, df: pd.DataFrame) -> pd.DataFrame:
        for c in df.columns:
            if c.startswith('t_') or c == 'time':
                df[c] /= self._n_runs
        return df
    
    def _aggregate_df(self, grouped_df, grouped_by, agg_funcs) -> pd.DataFrame:
        tmp_aggs = list(agg_funcs)
        p_index = self._get_index_and_pop(tmp_aggs, 'percentage')
        m_index = self._get_index_and_pop(tmp_aggs, 'single')
        sum_flag = False
        if m_index != -1 and 'count' not in tmp_aggs:
            tmp_aggs = tmp_aggs + ['count']
        if (p_index != -1 or m_index != -1) and 'sum' not in tmp_aggs:
            sum_flag = True
            tmp_aggs = tmp_aggs + ['sum']
        df = self._pd_funcs_agg_df(grouped_df, grouped_by, tmp_aggs)
        if m_index != -1:
            df.insert(m_index + 1, 't_single', df['t_sum'] / df['count'])
        if p_index != -1:
            total_time = df['t_sum'].sum()
            df.insert(p_index + 1, '%_of_total', df['t_sum'] * 100 / total_time)
        if 'count' not in agg_funcs and m_index != -1:
            df.drop('count', inplace=True, axis=1)
        if 'sum' not in agg_funcs and sum_flag:
            df.drop('t_sum', inplace=True, axis=1)
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
        if not all(a in self._ALLOWED_AGG_FUNCS for a in agg_funcs):
            raise ValueError("Invalid aggregate function"
                             "You should use one of these:"
                             f"{self._ALLOWED_AGG_FUNCS}")
        
    @abstractmethod
    def make_report(self, group_by, agg_funcs, sort_by, normilize=True) -> pd.DataFrame:
        if agg_funcs == None or agg_funcs == []:
            agg_funcs = self._DEFAULT_AGG_FUNCS
        self._check_report_capability(group_by, agg_funcs)
        report = self._estimations_table.copy()
        if group_by == None:
            report.sort_values(sort_by or 'time', ascending=False,
                               ignore_index=True, inplace=True)
        else:
            grouped = report.groupby(group_by, as_index=False)
            report = self._aggregate_df(grouped, group_by, agg_funcs)
            if sort_by == None:
                sort_by = agg_funcs[0]
            if sort_by != 'count' and sort_by in self._ALLOWED_AGG_FUNCS:
                sort_by = "t_" + sort_by
            report.sort_values(sort_by, ascending=False,
                               ignore_index=True, inplace=True)
        if normilize:
            return self._normilize(report)
        return report
    
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


class IndividualProfiling(Profiling):
    _n_runs: int
    _estimations_table: pd.DataFrame

    DEFAULT_RUNS = 10000
    _TABLE_COLUMNS = ("node", "type", "name", "time")
    _ALLOWED_GROUPBY = ("node", "type", "name")
    
    def __init__(self,
                 target_nodes: Sequence[FunctionNode]=[],
                 *,
                 source: Sequence[FunctionNode]=[],
                 sink: Sequence[FunctionNode]=[],
                 n_runs: int=DEFAULT_RUNS):
        super().__init__(target_nodes, source, sink, n_runs)

    @classmethod
    def estimate_node(cls, node: FunctionNode, n_runs: int=DEFAULT_RUNS):
        for input in node.inputs.iter_all():
            input.touch()
        
        return timeit(stmt=node.fcn, number=n_runs)

    def estimate_target_nodes(self) -> IndividualProfiling:
        records = {col: [] for col in self._TABLE_COLUMNS}
        for node in self._target_nodes:
            estimations = self.estimate_node(node, self._n_runs)
            records["node"].append(node)
            records["type"].append(type(node).__name__)
            records["name"].append(node.name)
            records["time"].append(estimations)
        self._estimations_table = pd.DataFrame(records)
        return self

    def make_report(self,
                    group_by: str | None="type",
                    agg_funcs: Sequence[str] | None=None,
                    sort_by: str | None=None):
        return super().make_report(group_by, agg_funcs, sort_by)

    def print_report(self, 
                     rows: int | None=10,
                     group_by: str | None="type",
                     agg_funcs: Sequence[str] | None=None,
                     sort_by: str | None=None):
        report = self.make_report(group_by, agg_funcs, sort_by)
        print(f"\nIndividual Profilng {hex(id(self))}, "
              f"n_runs for each node: {self._n_runs}\n"
              f"sort by: {sort_by or 'default sorting'}, "
              f"max rows displayed: {rows}")
        super()._print_table(report, rows)
        self._print_total_time()
        


class FrameworkProfiling(Profiling):
    _ALLOWED_GROUPBY = [["source nodes", "sink nodes"],  # [a, b] - group by two
                       "source nodes",                   #  columns a and b
                       "sink nodes"]

    def __init__(self,
                 target_nodes: Sequence[FunctionNode]=[],
                 *,
                 source: Sequence[FunctionNode]=[],
                 sink: Sequence[FunctionNode]=[],
                 n_runs = 100) -> None:
        super().__init__(target_nodes, source, sink, n_runs)
        if not self._source or not self._sink:
            self._reveal_source_sink()

    def _taint_nodes(self):
        for node in self._target_nodes:
            node.taint()

    @staticmethod
    def fcn_no_computation(node: FunctionNode):
        for input in node.inputs.iter_all():
            input.touch()

    def _make_fcns_empty(self):
        for node in self._target_nodes:
            node._stash_fcn()
            node.fcn = types.MethodType(self.fcn_no_computation, node)

    def _restore_fcns(self):
        for node in self._target_nodes:
            node._unwrap_fcn()

    def _estimate_framework_time(self) -> list[float]:
        self._make_fcns_empty()
        setup = lambda: self._taint_nodes()
        def repeat_stmt():
            for sink_node in self._sink:
                sink_node.eval()
        results = repeat(stmt=repeat_stmt, setup=setup,
                         repeat=self._n_runs, number=1)
        self._restore_fcns()
        return results

    def estimate_framework_time(self, 
                                append_results: bool=False) -> FrameworkProfiling:
        results = self._estimate_framework_time()
        df = pd.DataFrame(results, columns=["time"])
        sink_names = str([n.name for n in self._sink])
        source_names = str([n.name for n in self._source])
        df.insert(0, "sink nodes", shorten(sink_names, width=SINK_COL_WIDTH))
        df.insert(0, "source nodes", shorten(source_names, width=SOURCE_COL_WIDTH))
        if append_results and hasattr(self, "_estimations_table"):
            self._estimations_table = pd.concat([self._estimations_table, df])
        else:
            self._estimations_table = df
        return self

    def make_report(self,
                    group_by=["source nodes", "sink nodes"],
                    agg_funcs: Sequence[str] | None=None,
                    sort_by: str | None=None):
        return super().make_report(group_by, agg_funcs, sort_by)

    def print_report(self, 
                     rows: int | None=10,
                     group_by=["source nodes", "sink nodes"],
                     agg_funcs: Sequence[str] | None=None,
                     sort_by: str | None=None):
        report = self.make_report(group_by, agg_funcs, sort_by)
        print(f"\nFramework Profling {hex(id(self))}, "
              f"n_runs for given subgraph: {self._n_runs}, "
              f"nodes in subgraph: {len(self._target_nodes)}\n"
              f"sort by: `{sort_by or 'default sorting'}`, "
              f"max rows displayed: {rows}")
        return super()._print_table(report, rows)
