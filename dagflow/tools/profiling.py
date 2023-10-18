from __future__ import annotations

from timeit import timeit, repeat
import collections
from collections.abc import Generator
from typing import List, Set

import numpy as np
import pandas as pd
import tabulate

from ..nodes import FunctionNode


class ProfilingBase:
    _target_nodes: List[FunctionNode]
    _estimations_table: pd.DataFrame
    _ALLOWED_GROUPBY: list

    def __init__(self, target_nodes):
        self._target_nodes = target_nodes

    def check_report_capability(self, group_by):
        if not hasattr(self, "_estimations_table"):
            raise ValueError("No estimations found!\n"
                             "Note: first esimate your nodes "
                             "with methods like `estimate_*`")
        if group_by != None and group_by not in self._ALLOWED_GROUPBY:
            raise ValueError(f"Invalid `group_by` name \"{group_by}\"."
                             f"You must use one of these: {self._ALLOWED_GROUPBY}")
        
    def _print_report(self, data_frame, rows):
        print(tabulate.tabulate(data_frame.head(rows), 
                                headers='keys', 
                                tablefmt='psql'))
        
    def make_report(self, top_n=10, group_by=None):
        self.check_report_capability(group_by)
        self._print_report(self._estimations_table, top_n)
        raise NotImplementedError

class IndividualProfiling(ProfilingBase):
    _n_runs: int
    _estimations_table: pd.DataFrame

    DEFAULT_RUNS = 10000
    _TABLE_COLUMNS = ["node",
                      "type",
                      "name",
                      "time"]
    _ALLOWED_GROUPBY = ["node", "type", "name"]
    
    def __init__(self,
                 target_nodes: List[FunctionNode],
                 n_runs: int=DEFAULT_RUNS):
        super().__init__(target_nodes)
        self._n_runs = n_runs

    @classmethod
    def estimate_node(cls, node: FunctionNode, n_runs: int=DEFAULT_RUNS):
        for input in node.inputs.iter_all():
            input.touch()
        
        testing_function = lambda : node.fcn(node, node.inputs, node.outputs)
        return timeit(stmt=testing_function, number=n_runs)

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
    
    def _print_report(self, data_frame, rows):
        print(f"\nIndividual Profilng {hex(id(self))}, "
              f"n_runs for each node: {self._n_runs}, "
              f"max rows displayed: {rows}")
        return super()._print_report(data_frame, rows)

    def _aggregate_df(self, grouped_df) -> pd.DataFrame:
        df = grouped_df.agg({'time': 
                             ['count', 'mean', 'median', 
                              'std', 'min', 'max']})
        # get rid of multiindex and add prefix `t_` - time notation
        new_columns = ['type', 'count']
        new_columns += ['t_' + c[1] for c in df.columns[2:]]
        df.columns = new_columns
        return df
    
    def make_report(self, 
                    top_n: int | None=10,
                    group_by: str | None="type"):
        super().check_report_capability(group_by)
        if group_by == None:
            report = self._estimations_table.sort_values("time",
                                                         ascending=False,
                                                         ignore_index=True)
        else:
            grouped = self._estimations_table.groupby(group_by, as_index=False)
            report = self._aggregate_df(grouped)
            report.sort_values("t_mean", ascending=False, ignore_index=True,
                                inplace=True)
        self._print_report(report, top_n)
    
class GroupProfiling(ProfilingBase):

    # _estimations: List[EstimateRecord]
    # _excluded_nodes: List[FunctionNode]
    _target_nodes: List[FunctionNode]
    _source: List[FunctionNode]
    _sink: List[FunctionNode]
    _n_runs: int

    _ALLOWED_GROUPBY = [["source nodes", "sink nodes"],  # groupby=[a, b] - groub  
                       "source nodes",                  #  by two columns a and b
                       "sink nodes"]
    # TODO: align with the parent class
    DEFAULT_AGG_FUNCS = ['count', 'mean', 'median', 'std', 'min', 'max']

    def __init__(self, 
                 source: List[FunctionNode]=[],
                 sink: List[FunctionNode]=[],
                 n_runs = 100) -> None:
        # self._estimations = list()
        # self._excluded_nodes = excluded_nodes
        self._source = source
        self._sink = sink
        self._n_runs = n_runs
        self._target_nodes = list(self._gather_related_nodes())

    def __child_nodes_gen(self, node: FunctionNode) -> Generator[FunctionNode, None, None]:
        for output in node.outputs.iter_all():
            for child_input in output.child_inputs:
                yield child_input.node

    def __check_reachable(self, nodes_gathered):
        if not all(s in nodes_gathered for s in self._sink):
            raise ValueError("Some of the `sink` nodes are unreachable "
                             "(no paths from source)")

    def _gather_related_nodes(self) -> Set[FunctionNode]:
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

    def _taint_nodes(self):
        for node in self._target_nodes:
            node.taint()

    # using only `node` argument for further compatibility
    @staticmethod
    def fcn_no_computation(node: FunctionNode, inputs, outputs):
        for input in node.inputs.iter_all():
            input.touch()

    def _make_fcns_empty(self):
        for node in self._target_nodes:
            node._stash_fcn()
            node.fcn = self.fcn_no_computation

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

    def estimate_framework_time(self, append_results: bool=False):
        results = self._estimate_framework_time()
        df = pd.DataFrame(results, columns=["time"])
        df.insert(0, "sink nodes", str([n.name for n in self._sink]))
        df.insert(0, "source nodes", str([n.name for n in self._source]))
        if append_results and hasattr(self, "_estimations_table"):
            self._estimations_table = pd.concat([self._estimations_table, df])
        else:
            self._estimations_table = df

    def _aggregate_df(self, grouped_df, grouped_by) -> pd.DataFrame:
        df = grouped_df.agg({'time': self.DEFAULT_AGG_FUNCS})
        # grouped_by can be ["col1", "col2", ...] or "col"
        new_columns = grouped_by if type(grouped_by)==list else [grouped_by]
        # get rid of multiindex and add prefix `t_` - time notation
        new_columns += ['count']
        new_columns += ['t_' + c for c in self.DEFAULT_AGG_FUNCS[1:]]
        df.columns = new_columns
        return df
    
    def _print_report(self, data_frame, rows):
        print(f"\nGroup Profling {hex(id(self))}, "
              f"n_runs for each estimation: {self._n_runs}, "
              f"nodes in group: {len(self._target_nodes)}, "
              f"max rows displayed: {rows}")
        return super()._print_report(data_frame, rows)


    def make_report(self,
                    top_n: int | None=10,
                    group_by=["source nodes", "sink nodes"]):
        super().check_report_capability(group_by)
        if group_by == None:
            report = self._estimations_table.sort_values("time",
                                                         ascending=False,
                                                         ignore_index=True)
        else:
            grouped = self._estimations_table.groupby(group_by, as_index=False)
            report = self._aggregate_df(grouped, group_by)
            report.sort_values("t_mean", ascending=False, ignore_index=True,
                                inplace=True)
        self._print_report(report, top_n)
