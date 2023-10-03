from __future__ import annotations

from timeit import timeit, repeat
from functools import cached_property
import collections
from typing import List

import numpy as np
import pandas as pd
import tabulate

from ..nodes import FunctionNode


# depricated
# TODO: remove this class
class EstimateRecord:

    _node: FunctionNode
    _n_runs: int
    _total_time: float

    # It is weird to use __slots__ with __dict__,  
    # but it works slightly faster than without it. 
    # __dict__ is required for @functools.cached_property 
    __slots__ = ("_node", "_n_runs", "_total_time", "__dict__")

    def __init__(self, 
                 node: FunctionNode, 
                 n_runs: int, 
                 estimated_time: float):
        self._node = node
        self._n_runs = n_runs
        self._total_time = estimated_time

    @property
    def node_name(self):
        return self._node.name
    
    @property
    def type(self):
        return type(self._node).__name__
    
    @property
    def n_runs(self):
        return self._n_runs
    
    @property
    def time(self):
        return self._total_time
    
    @cached_property
    def avg_time(self):
        if self._n_runs > 0:
            return self._total_time / self._n_runs
        return 0
    
    def __lt__(self, other):
        return self.avg_time < other.avg_time
    
    def __str__(self) -> str:
        return f"name={self.node_name}, runs={self._n_runs}, avg={self.avg_time}"
    

class ProfilingBase:
    _target_nodes: List[FunctionNode]
    _estimations_table: pd.DataFrame

    def __init__(self, target_nodes):
        self._target_nodes = target_nodes


class IndividualProfiling(ProfilingBase):
    _n_runs: int
    _estimations_table: pd.DataFrame

    DEFAULT_RUNS = 10000
    _TABLE_COLUMNS = ["node",
                      "type",
                      "name",
                      "time"]
    
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
        print(f"\nFirst {rows} rows of profiling report,",
              f"where n_runs={self._n_runs} for each node:")
        ## print using `pandas.option_context``:
        # with pd.option_context("display.max_rows", None,
        #                        "display.max_columns", None,
        #                        "display.precision", 6):
        #     print(data_frame.head(rows)) 
        
        # print with tabulte
        print(tabulate.tabulate(data_frame.head(rows), 
                                headers='keys', 
                                tablefmt='psql'))

    def _aggregate_df(self, grouped_df) -> pd.DataFrame:
        df = grouped_df.agg({'time': 
                             ['count', 'mean', 'median', 
                              'std', 'min', 'max']})
        # remove column multiindex and add prefix `t_` - time notation
        new_columns = ['type', 'count']
        new_columns += ['t_' + c[1] for c in df.columns[2:]]
        df.columns = new_columns
        return df
    
    def make_report(self, 
                    top_n=10,
                    group_by: str | None="type"):
        if hasattr(self, "_estimations_table"):
            if group_by == None:
                report = self._estimations_table.sort_values("time",
                                                             ascending=False)
            elif group_by in self._TABLE_COLUMNS:
                grouped = self._estimations_table.groupby(group_by, 
                                                          as_index=False)
                report = self._aggregate_df(grouped)
                report.sort_values("t_mean", ascending=False, inplace=True)
            else:
                raise ValueError(f"Invalid `group_by` name \"{group_by}\"."
                                 f"You must use one of these: {self._TABLE_COLUMNS}")
            self._print_report(report, top_n)
        else:
            # TODO: check dagflow errors
            raise ValueError("No estimations found!\n"
                             "Hint: use `estimate_target_nodes`"
                             "to individually estimate group of nodes")
        
    
class GroupProfiling:

    _estimations: List[EstimateRecord]
    # _excluded_nodes: List[FunctionNode]
    _target_nodes: List[FunctionNode]
    _removed_fcns: collections.deque[FunctionNode]
    _n_runs: int

    def __init__(self, 
                 targen_nodes: List[FunctionNode] = [],
                 n_runs = 1) -> None:
        self._estimations = list()
        # self._excluded_nodes = excluded_nodes
        self._target_nodes = targen_nodes
        self._n_runs = n_runs
        self._removed_fcns = collections.deque()

    def taint_parents(self, node): 
        if node not in self._excluded_nodes:
            for input in node.inputs.iter_all():
                    self.taint_parents(input.parent_node)

            node.taint()

    # using only `node` argument for further compatibility
    @staticmethod
    def fcn_no_computation(node: FunctionNode, inputs, outputs):
        for input in node.inputs.iter_all():
            input.touch()

        # print(node.name)
        
        # Note: may work much slower than previos fcn() for nodes 
        # where return is just a number and etc
        # return list(node.outputs.iter_data())
        # return None
        

    def _make_fcns_empty(self, node: FunctionNode):
        for node in self._target_nodes:
            node._stash_fcn()
            node.fcn = self.fcn_no_computation()

        # if node not in self._excluded_nodes:
        #     for input in node.inputs.iter_all():
        #         self._make_fcns_empty(input.parent_node)

        #     self._removed_fcns.append(node.fcn)
        #     node.fcn = self.fcn_no_computation

    def _restore_fcns(self, node: FunctionNode):
        for node in self._target_nodes:
            node._unwrap_fcn()

        # if node not in self._excluded_nodes:
        #     for input in node.inputs.iter_all():
        #         self._restore_fcns(input.parent_node)

        #     node.fcn = self._removed_fcns.popleft()

    def estimate_group_with_empty_fcn(self, head_node: FunctionNode):
        raise NotImplementedError("this class doesn't work at all yet")
        self._make_fcns_empty(head_node)
        setup = lambda: self.taint_parents(head_node)

        results = np.array(repeat(stmt=head_node.eval, setup=setup, repeat=500, number=1))

        print("\tMean:", results.mean())
        print("\tStd:", results.std())
        print("\tMin:", results.min())

        self._restore_fcns(head_node)


