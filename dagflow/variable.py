# from numpy import array, copyto, result_type

from .node import Node
from .output import Output

class Parameter(object):
    # value = None
    _value_node: Node

    def __init__(self, value_node: Node):
        self._value_node = value_node

    @staticmethod
    def from_output(output: Output) -> Parameter:
        pass

class GaussianParameter(Parameter):
    # mean = None
    # sigma = None

    _mean_node: Node
    _sigma_node: Node

    def __init__(self, value_node: Node, mean_node: Node, sigma_node: Node):
        super().__init__(value_node)
        self._mean_node = mean_node
        self._sigma_node = sigma_node

    @staticmethod
    def from_outputs(value_output: Output, mean_output: Output, sigma_output: Output) -> GaussianParameter:
        pass
