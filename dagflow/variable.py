from .node import Node, Output
from .exception import InitializationError
from .lib.NormalizeCorrelatedVars2 import NormalizeCorrelatedVars2
from .lib.Cholesky import Cholesky
from .lib.Array import Array
from .lib.CovmatrixFromCormatrix import CovmatrixFromCormatrix

from numpy import zeros_like, array
from numpy.typing import DTypeLike
from typing import Optional, Dict

class Parameters(object):
    value: Output
    _value_node: Node

    def __init__(self, value: Node):
        self._value_node = value
        self.value = value.outputs[0]

    @staticmethod
    def from_numbers(*, dtype: DTypeLike='d', **kwargs) -> 'Parameters':
        sigma = kwargs['sigma']
        if sigma is not None:
            return GaussianParameters.from_numbers(dtype=dtype, **kwargs)

        label: Dict[str, str] = kwargs.get('label')
        if label is None:
            label = {'text': 'parameter'}
        else:
            label = dict(label)
        name: str = label.setdefault('name', 'parameter')
        value = kwargs['value']
        return Parameters(
            Array(
                name,
                array((value,), dtype=dtype),
                label = label,
                mode='store_weak'
            )
        )

class GaussianParameters(Parameters):
    central: Output
    sigma: Output
    normvalue: Output

    _central_node: Node
    _sigma_node: Node
    _normvalue_node: Node

    _cholesky_node: Optional[Node] = None
    _covariance_node: Optional[Node] = None
    _correlation_node: Optional[Node] = None
    _sigma_total_node: Optional[Node] = None

    _forward_node: Node
    _backward_node: Node

    def __init__(self, value: Node, central: Node, *, sigma: Node=None, covariance: Node=None, correlation: Node=None):
        super().__init__(value)
        self._central_node = central

        if sigma is not None and covariance is not None:
            raise InitializationError('GaussianParameters: got both "sigma" and "covariance" as arguments')
        if correlation is not None and sigma is None:
            raise InitializationError('GaussianParameters: got "correlation", but no "sigma" as arguments')

        if correlation is not None:
            self._correlation_node = correlation
            self._covariance_node = CovmatrixFromCormatrix(f"V({value.name})")
            self._cholesky_node = Cholesky(f"L({value.name})")
            self._sigma_total_node = sigma
            self._sigma_node = self._cholesky_node

            self._sigma_total_node >> self._covariance_node.inputs['sigma']
            correlation >> self._covariance_node
            self._covariance_node >> self._cholesky_node
        elif sigma is not None:
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
        self.normvalue = self._normvalue_node.outputs[0]

        self._norm_node = NormalizeCorrelatedVars2(f"Normalize {value.name}", immediate=True)
        self.central >> self._norm_node.inputs['central']
        self.sigma >> self._norm_node.inputs['matrix']
        (self.value, self.normvalue) >> self._norm_node

        self._norm_node.close(recursive=True)
        self._norm_node.touch()

    @staticmethod
    def from_numbers(
        value: float,
        *,
        central: float,
        sigma: float,
        label: Optional[Dict[str,str]]=None,
        dtype: DTypeLike='d',
        **_
    ) -> 'GaussianParameters':
        if label is None:
            label = {'text': 'gaussian parameter'}
        else:
            label = dict(label)
        name = label.setdefault('name', 'parameter')
        node_value = Array(
            name,
            array((value,), dtype=dtype),
            label = label,
            mode='store_weak'
        )

        node_central = Array(
            f'{name}_central',
            array((central,), dtype=dtype),
            label = {k: f'central: {v}' for k,v in label.items()},
            mode='store_weak'
        )

        node_sigma = Array(
            f'{name}_sigma',
            array((sigma,), dtype=dtype),
            label = {k: f'sigma: {v}' for k,v in label.items()},
            mode='store_weak'
        )

        return GaussianParameters(value=node_value, central=node_central, sigma=node_sigma)
