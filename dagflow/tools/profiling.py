from timeit import timeit
from functools import cached_property
from typing import List

from ..nodes import FunctionNode

class EstimateRecord:

    _node: FunctionNode
    _exec_times: int
    _total_time: float

    # It is weird to use __slots__ with __dict__,  
    # but it works slightly faster than without it. 
    # __dict__ is required for @functools.cached_property 
    __slots__ = ("_node", "_exec_times", "_total_time", "__dict__")

    def __init__(self, 
                 node: FunctionNode, 
                 exec_times: int, 
                 estimated_time: float):
        self._node = node
        self._exec_times = exec_times
        self._total_time = estimated_time

    @property
    def node_name(self):
        return self._node.name
    
    @property
    def type(self):
        return type(self._node).__name__
    
    # n runs
    @property
    def exec_times(self):
        return self._exec_times
    
    @property
    def time(self):
        return self._total_time
    
    @cached_property
    def avg_time(self):
        if self._exec_times > 0:
            return self._total_time / self._exec_times
        return 0
    
    def __lt__(self, other):
        return self.avg_time < other.avg_time


class Profiling:
    _exec_times: int
    _results: List[EstimateRecord]
    __slots__ = ("_exec_times", "_results")

    def __init__(self, exec_times: int=100):
        self._exec_times = exec_times
        self._results = []

    def estimate_node(self, node: FunctionNode, times: int=None):
        if not times:
            times = self._exec_times

        # check node inputs 
        for input in node.inputs:
            if input.tainted:
                input.touch()
        # node.touch(force=True)

        estimations = timeit(stmt=node._eval, number=times)

        # Maybe it's more appropriate to user node._fcn, 
        # because _fcn performs only related to calculations stuff
        # testing_function = lambda : node._fcn(node, node.inputs, node.outputs)
        # estimations = timeit(stmt=testing_function, number=times)

        result = EstimateRecord(node, times, estimations)

        # result = {
        #     "name": node.name, 
        #     "type": type(node).__name__, 
        #     "exec_times": times,
        #     "total_time": estimations
        # }

        self._results.append(result)
        return result
    
    def make_report(self, top_n=10):
        self._results.sort(reverse=True)
        print(f"\nTop {top_n} operations")
        print('=' * 93)
        line_format = "%-3s %-25s %-25s %-25s %11s" 
        print(line_format % ('#',
                             'Operation type',
                             'Name', 
                             'Average time',
                             'Exec. times'))
        print('-' * 93)
        for i, record in enumerate(self._results):
            print(line_format % (i,
                                 record.type,
                                 record.node_name,
                                 record.avg_time,
                                 record.exec_times))
        
