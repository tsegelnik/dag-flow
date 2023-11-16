from __future__ import annotations

from timeit import timeit, repeat
import collections
from collections.abc import Generator, Sequence
from abc import ABCMeta, abstractmethod

import pandas as pd
import tabulate

from ..nodes import FunctionNode

# abc - Abstract Base Class
class Profiling(metaclass=ABCMeta):
    _target_nodes: Sequence[FunctionNode]
    _source: Sequence[FunctionNode]
    _sink: Sequence[FunctionNode]
    _n_runs: int
    _estimations_table: pd.DataFrame
    # TODO: check for _ALLOWED_GROUPBY existence
    _ALLOWED_GROUPBY: tuple[str]
    _ALLOWED_AGG_FUNCS: tuple[str] = ("count", "mean", "median",
                                      "std", "min", "max")
    _DEFAULT_AGG_FUNCS: tuple[str] = ("min", "mean", "count")

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
    
    def _aggregate_df(self, grouped_df, grouped_by, agg_funcs) -> pd.DataFrame:
        df = grouped_df.agg({'time': agg_funcs})
        # grouped_by can be ["col1", "col2", ...] or "col"
        new_columns = grouped_by if type(grouped_by)==list else [grouped_by]
        # get rid of multiindex and add prefix `t_` - time notation
        new_columns += ['t_' + c if c != 'count' else 'count' for c in agg_funcs]
        df.columns = new_columns
        return df
    
    def _check_report_capability(self, group_by, agg_funcs):
        if not hasattr(self, "_estimations_table"):
            raise AttributeError("No estimations found!\n"
                                 "Note: first esimate your nodes "
                                 "with methods like `estimate_*`")
        if group_by != None and group_by not in self._ALLOWED_GROUPBY:
            raise ValueError(f"Invalid `group_by` name \"{group_by}\"."
                             f"You must use one of these: {self._ALLOWED_GROUPBY}")
        if not all(a in self._ALLOWED_AGG_FUNCS for a in agg_funcs):
            raise ValueError("Invalid aggregate function"
                             "You should use one of these:"
                             f"{self._ALLOWED_AGG_FUNCS}")
        
    @abstractmethod
    def make_report(self, group_by, agg_funcs, sort_by) -> pd.DataFrame:
        if agg_funcs == None or agg_funcs == []:
            agg_funcs = self._DEFAULT_AGG_FUNCS
        if sort_by != 'count' and sort_by in self._ALLOWED_AGG_FUNCS:
            sort_by = "t_" + sort_by
        self._check_report_capability(group_by, agg_funcs)
        if group_by == None:
            report = self._estimations_table.sort_values(sort_by or 'time',
                                                         ascending=False,
                                                         ignore_index=True)
        else:
            grouped = self._estimations_table.groupby(group_by, as_index=False)
            report = self._aggregate_df(grouped, group_by, agg_funcs)
            # TODO: add check for agg_funcs[0] == 'count'
            report.sort_values(sort_by or 't_' + agg_funcs[0], ascending=False, ignore_index=True,
                               inplace=True)
        return report
    
    def _print_table(self, dataframe, rows):
        print(tabulate.tabulate(dataframe.head(rows), 
                                headers='keys', 
                                tablefmt='psql'))
    
    @abstractmethod
    def print_report(self, rows, *args, **kwargs):
        report = self.make_report(rows, *args, **kwargs)
        self._print_table(report)
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
              f"n_runs for each node: {self._n_runs}, "
              f"max rows displayed: {rows}")
        return super()._print_table(report, rows)
    

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

    def estimate_framework_time(self, 
                                append_results: bool=False) -> FrameworkProfiling:
        results = self._estimate_framework_time()
        df = pd.DataFrame(results, columns=["time"])
        df.insert(0, "sink nodes", str([n.name for n in self._sink]))
        df.insert(0, "source nodes", str([n.name for n in self._source]))
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
              f"n_runs for each estimation: {self._n_runs}, "
              f"nodes in group: {len(self._target_nodes)}, "
              f"max rows displayed: {rows}")
        return super()._print_table(report, rows)
