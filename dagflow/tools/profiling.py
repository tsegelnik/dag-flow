from __future__ import annotations

from timeit import timeit, repeat
from functools import cached_property
import collections
from typing import List, Set
import numpy as np


from ..nodes import FunctionNode

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


class Profiling:
    _n_runs: int
    _results: List[EstimateRecord]
    __slots__ = ("_n_runs", "_results")

    def __init__(self, n_runs: int=100):
        self._n_runs = n_runs
        self._results = []

    def estimate_node(self, node: FunctionNode, n_runs: int=None):
        if not n_runs:
            n_runs = self._n_runs

        # compute and cache all inputs to prevent 
        # calculations during time measurment
        for input in node.inputs.iter_all():
            input.touch()

        testing_function = lambda : node.fcn(node, node.inputs, node.outputs)
        testing_function() # ignore the first calculation
        # estimations = timeit(stmt=testing_function, number=n_runs)
        estimations = repeat(stmt=testing_function, repeat=100, number=10000)
        # print(estimations)
        print(node.name, min(estimations), max(estimations))

        result = EstimateRecord(node, n_runs, min(estimations))

        self._results.append(result)
        return result
    
    def estimate_graph(self, graph):
        for node in graph._nodes:
            self.estimate_node(node, n_runs=self._n_runs)

        return self
    
    def make_report(self, top_n=10):
        self._results.sort(reverse=True)
        print(f"\nTop {min(top_n, len(self._results))} operations")
        print('=' * 93)
        line_format = "%-3s %-25s %-25s %-25s %11s" 
        print(line_format % ('#',
                             'Operation type',
                             'Name', 
                             'Average time',
                             'Exec. runs'))
        print('-' * 93)
        for i, record in enumerate(self._results):
            print(line_format % (i + 1,
                                 record.type,
                                 record.node_name,
                                 record.avg_time,
                                 record.n_runs))
        
class GroupProfiling:

    _estimations: List[EstimateRecord]
    _excluded_nodes: List[FunctionNode]
    _removed_fcns: collections.deque[FunctionNode]
    _n_runs: int

    # slots

    # список узлов, которые нас интересуют

    # del node.fcn

    # _stash_ _unwrap_

    # make base class   

    def __init__(self, 
                 excluded_nodes: List[FunctionNode] = [],
                 n_runs = 1) -> None:
        self._estimations = list()
        self._excluded_nodes = excluded_nodes
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
        

    def make_fcns_empty(self, node: FunctionNode): 
        if node not in self._excluded_nodes:
            for input in node.inputs.iter_all():
                self.make_fcns_empty(input.parent_node)

            self._removed_fcns.append(node.fcn)
            node.fcn = self.fcn_no_computation

    def restore_fcns(self, node: FunctionNode):
        if node not in self._excluded_nodes:
            for input in node.inputs.iter_all():
                self.restore_fcns(input.parent_node)

            node.fcn = self._removed_fcns.popleft()

    def estimate_group_with_empty_fcn(self, head_node: FunctionNode):
        self.make_fcns_empty(head_node)
        setup = lambda: self.taint_parents(head_node)

        results = np.array(repeat(stmt=head_node.eval, setup=setup, repeat=500, number=1))

        print("\tMean:", results.mean())
        print("\tStd:", results.std())
        print("\tMin:", results.min())

        self.restore_fcns(head_node)


    # def estimate_group_with_normal_fcn(self, head_node: FunctionNode):
    #     self.taint_parents(head_node)
        



    # def estimate_group(self,    
    #                    head_node: FunctionNode, 
    #                    excluded_nodes: List[FunctionNode]):
    #     tree_stack = collections.deque()
    #     tree_stack.append(head_node)
    #     cur_node = head_node
    #     while tree_stack.count():
    #         for input in cur_node.inputs.iter_all():
    #         for head_node.inputs.iter_all()
    #         cur_node = head_node
    #         for input in head_node.inputs.iter_all():
        
        # полный обход дерева
        # вычисление fcn для каждого 
        # вычисление 1 раз? exept
        # вычисление времени когда все родительские taint, посчитать разницу
        # подумать как хранить данные
        



