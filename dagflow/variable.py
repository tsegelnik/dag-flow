# from numpy import array, copyto, result_type

from .node import Node

class Parameter(object):
    variable = None
    _variable_node: Node

    def __init__(self, variable_node: Node):
        self._variable_node = variable_node

class GaussianParameter(Parameter):
    mean = None
    sigma = None

    _mean_node: Node
    _sigma_node: Node

    def __init__(self, variable_node: Node, mean_node: Node, sigma_node: Node):
        super().__init__(variable_node)
        self._mean_node = mean_node
        self._sigma_node = sigma_node

