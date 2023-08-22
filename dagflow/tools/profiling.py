from timeit import timeit
from functools import cached_property
from typing import List

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

        # check node inputs 
        for input in node.inputs:
            if input.tainted:
                input.touch()
        # node.touch(force=True)

        testing_function = lambda : node._fcn(node, node.inputs, node.outputs)
        estimations = timeit(stmt=testing_function, number=n_runs)

        result = EstimateRecord(node, n_runs, estimations)

        self._results.append(result)
        return result
    
    def estimate_graph(self, graph):
        for node in graph._nodes:
            self.estimate_node(node, n_runs=10000)

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
        
