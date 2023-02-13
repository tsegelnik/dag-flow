# from numpy import array, copyto, result_type

from .node import Node, Output
from .exception import InitializationError
from .lib.NormalizeCorrelatedVars import NormalizeCorrelatedVars
from .lib.SharedInputsNode import SharedInputsNode
from .lib.Cholesky import Cholesky
from .lib.Array import Array

from numpy import zeros_like
from typing import Optional

class Parameters(object):
    value: Output
    _value_node: Node

    def __init__(self, value: Node):
        self._value_node = value
        self.value = value.outputs[0]

class GaussianParameters(Parameters):
    central: Output
    sigma: Output
    normvalue: Output

    _central_node: Node
    _sigma_node: Node
    _normvalue_node: Node

    _cholesky_node: Optional[Node] = None
    _covariance_node: Optional[Node] = None

    _forward_node: Node
    _backward_node: Node

    def __init__(self, value: Node, central: Node, *, sigma: Node=None, covariance: Node=None):
        super().__init__(value)
        self._central_node = central

        if sigma is not None and covariance is not None:
            raise InitializationError('GaussianParameters: got both "sigma" and "covariance" as arguments')

        if sigma is not None:
            self._sigma_node = sigma
        elif covariance is not None:
            self._cholesky_node = Cholesky(f"L({value.name})")
            self._sigma_node = self._cholesky_node
            self._covariance_node = covariance

            covariance >> self._cholesky_node
        else:
            # TODO: no sigma/covariance AND central means normalized=value?
            raise InitializationError('GaussianParameters: got no "sigma" and no "covariance" arguments')

        self.central = self._central_node.outputs[0]
        self.sigma = self._sigma_node.outputs[0]

        self._normvalue_node = Array(
            f'Normalized {value.name}',
            zeros_like(self.central._data),
            mark = f'norm({value.mark})',
            mode='store_weak'
        )

        #
        # Correlated → normalized
        #
        self._forward_node = NormalizeCorrelatedVars(f"Normalize {value.name}", mode='forward', immediate=True)
        self.central >> self._forward_node.inputs['central']
        self.sigma >> self._forward_node.inputs['matrix']
        self.value >> self._forward_node
        self.normvalue = self._normvalue_node.outputs[0]

        #
        # Normalized → correlated
        #
        self._backward_node = NormalizeCorrelatedVars(f"Unnormalize {value.name}", mode='backward', immediate=True)
        self.central >> self._backward_node.inputs['central']
        self.sigma >> self._backward_node.inputs['matrix']
        self._normvalue_node >> self._backward_node

        #
        # Shared nodes
        #
        self._common_value_node = SharedInputsNode(f'{value.name} (mid)')
        self._common_normvalue_node = SharedInputsNode(f'Normalized {value.name} (mid)')

        self.value >> self._common_value_node
        self._backward_node >> self._common_value_node

        self._normvalue_node >> self._common_normvalue_node
        self._forward_node >> self._common_normvalue_node

        # self._common_normvalue_node.update_types(recursive=True)
        # self._common_value_node.update_types(recursive=True)
        # self._common_normvalue_node.allocate(recursive=True)
        # self._common_value_node.allocate(recursive=True)
        self._common_normvalue_node.close(together=[self._common_value_node])
        self._common_normvalue_node.touch()
        self._common_value_node.touch()

