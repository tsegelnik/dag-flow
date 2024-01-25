from collections.abc import Sequence
from numpy import zeros_like, array, ndarray
from numpy.typing import DTypeLike, ArrayLike
from typing import Optional, Dict, Tuple, List, Union, Generator

from .node import Node, Output
from .exception import InitializationError
from .lib.NormalizeCorrelatedVars2 import NormalizeCorrelatedVars2
from .lib.Cholesky import Cholesky
from .lib.Array import Array
from .lib.CovmatrixFromCormatrix import CovmatrixFromCormatrix
from .labels import inherit_labels


class Parameter:
    __slots__ = ("_idx", "_parent", "_value_output", "_connectible", "_labelfmt")
    _parent: "Parameters"
    _idx: int
    _value_output: Output
    _connectible: Output
    _labelfmt: str

    def __init__(
        self,
        value_output: Output,
        idx: int = 0,
        *,
        parent: "Parameters",
        connectible: Optional[Output] = None,
        labelfmt: str = "{}",
    ):
        self._idx = idx
        self._parent = parent
        self._value_output = value_output
        self._labelfmt = labelfmt
        self._connectible = value_output if connectible is None else connectible

    def __str__(self):
        return f"par v={self.value}"

    @property
    def value(self) -> float:
        return self._value_output.data[self._idx]

    @value.setter
    def value(self, value: float):
        return self._value_output.seti(self._idx, value)

    @property
    def output(self) -> Output:
        return self._value_output

    @property
    def is_correlated(self) -> bool:
        return self._parent.is_correlated

    @property
    def connectible(self) -> Output:
        return self._connectible

    def label(self, source: str = "text") -> str:
        return self._labelfmt.format(self._value_output.node.labels[source])

    def to_dict(self, *, label_from: str = "text") -> dict:
        return {"value": self.value, "label": self.label(label_from), "flags": ""}

    def __rshift__(self, other):
        self._connectible >> other


class GaussianParameter(Parameter):
    __slots__ = ("_central_output", "_sigma_output", "_normvalue_output")
    _central_output: Output
    _sigma_output: Output
    _normvalue_output: Output

    def __init__(
        self,
        value_output: Output,
        central_output: Output,
        sigma_output: Output,
        idx: int = 0,
        *,
        normvalue_output: Output,
        **kwargs,
    ):
        super().__init__(value_output, idx, **kwargs)
        self._central_output = central_output
        self._sigma_output = sigma_output
        self._normvalue_output = normvalue_output

    def __str__(self):
        self.central
        if self.central != 0:
            return (
                f"gpar v={self.value} ({self.normvalue}σ):"
                f" {self.central}±{self.sigma} ({self.sigma_percent:.2g}%)"
            )
        else:
            return f"gpar v={self.value} ({self.normvalue}σ): {self.central}±{self.sigma}"

    @property
    def central(self) -> float:
        return self._central_output.data[0]

    @central.setter
    def central(self, central: float):
        self._central_output.seti(self._idx, central)

    @property
    def sigma(self) -> float:
        return self._sigma_output.data[0]

    @sigma.setter
    def sigma(self, sigma: float):
        self._sigma_output.seti(self._idx, sigma)

    @property
    def sigma_relative(self) -> float:
        return self.sigma / self.value

    @sigma_relative.setter
    def sigma_relative(self, sigma_relative: float):
        self.sigma = sigma_relative * self.value

    @property
    def sigma_percent(self) -> float:
        return 100.0 * (self.sigma / self.value)

    @sigma_percent.setter
    def sigma_percent(self, sigma_percent: float):
        self.sigma = (0.01 * sigma_percent) * self.value

    @property
    def normvalue(self) -> float:
        return self._normvalue_output.data[0]

    @normvalue.setter
    def normvalue(self, normvalue: float):
        self._normvalue_output.seti(self._idx, normvalue)

    def to_dict(self, **kwargs) -> dict:
        dct = super().to_dict(**kwargs)
        dct.update(
            {
                "central": self.central,
                "sigma": self.sigma,
                # 'normvalue': self.normvalue,
            }
        )
        if self.is_correlated:
            dct["flags"] += "C"
        return dct


class NormalizedGaussianParameter(Parameter):
    @property
    def central(self) -> float:
        return 0.0

    @property
    def sigma(self) -> float:
        return 1.0

    def to_dict(self, **kwargs) -> dict:
        dct = super().to_dict(**kwargs)
        dct.update(
            {
                "central": 0.0,
                "sigma": 1.0,
            }
        )
        return dct


class Constraint:
    __slots__ = ("_pars",)
    _pars: "Parameters"

    def __init__(self, parameters: "Parameters"):
        self._pars = parameters

    @property
    def is_correlated(self) -> bool:
        return False


class Parameters:
    __slots__ = (
        "value",
        "_value_node",
        "_pars",
        "_names",
        "_norm_pars",
        "_is_variable",
        "_constraint",
    )
    value: Output
    _value_node: Node
    _pars: List[Parameter]
    _names: Tuple[Tuple[str, ...], ...]
    _norm_pars: List[Parameter]

    _is_variable: bool

    _constraint: Optional[Constraint]

    def __init__(
        self,
        names: Tuple[Tuple[str, ...], ...],
        value: Node,
        *,
        variable: Optional[bool] = None,
        fixed: Optional[bool] = None,
        close: bool = True,
    ):
        self._value_node = value
        try:
            self.value = value.outputs[0]
        except IndexError:
            raise InitializationError("Parameters: value node has no outputs")

        if all(f is not None for f in (variable, fixed)):
            raise RuntimeError("Parameter may not be set to variable and fixed at the same time")
        if variable is not None:
            self._is_variable = variable
        elif fixed is not None:
            self._is_variable = not fixed
        else:
            self._is_variable = True

        self._constraint = None

        self._pars = []
        self._names = tuple((name,) if isinstance(name, str) else name for name in names)
        self._norm_pars = []
        if close:
            self._close()

            self._pars.extend(
                Parameter(self.value, i, parent=self) for i in range(self.value._data.size)
            )

    def _close(self) -> None:
        self._value_node.close(recursive=True)

    @property
    def is_variable(self) -> bool:
        return self._is_variable

    @property
    def is_fixed(self) -> bool:
        return not self._is_variable

    @property
    def is_constrained(self) -> bool:
        return self._constraint is not None

    @property
    def is_free(self) -> bool:
        return self._constraint is None

    @property
    def is_correlated(self) -> bool:
        return False if self._constraint is None else self._constraint.is_correlated

    @property
    def parameters(self) -> List:
        return self._pars

    @property
    def norm_parameters(self) -> List:
        return self._norm_pars

    def iteritems(self) -> Generator[Tuple[Tuple[str, ...], Parameter], None, None]:
        yield from zip(self._names, self._pars)

    def iteritems_norm(self) -> Generator[Tuple[Tuple[str, ...], Parameter], None, None]:
        yield from zip(self._names, self._norm_pars)

    def _reset_pars(self) -> None:
        self._pars = []
        self._norm_pars = []

    @property
    def constraint(self) -> Optional[Constraint]:
        return self._constraint

    def to_dict(self, *, label_from: str = "text") -> dict:
        return {
            "value": self.value.data[0],
            "label": self._value_node.labels[label_from],
            "flags": "",
        }

    def set_constraint(self, constraint: Constraint) -> None:
        if self._constraint is not None:
            raise InitializationError("Constraint already set")
        self._constraint = constraint
        # constraint._pars = self

    @staticmethod
    def from_numbers(
        value: Union[float, int, ArrayLike],
        *,
        names: Tuple[Tuple[str, ...], ...] = ((),),
        dtype: DTypeLike = "d",
        variable: Optional[bool] = None,
        fixed: Optional[bool] = None,
        label: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> "Parameters":
        if label is None:
            label = {"text": "parameter"}
        else:
            label = dict(label)
        name: str = label.setdefault("name", "parameter")
        has_constraint = kwargs.get("sigma", None) is not None

        if isinstance(value, (float, int)):
            value = (value,)
        elif not isinstance(value, (Sequence, ndarray)):
            raise InitializationError(
                f"Parameters.from_numbers: Unsupported value type {type(value)}"
            )
        if len(names) != len(value):
            raise InitializationError(
                f"Parameters.from_numbers: inconsistent values ({value}) and names ({names})"
            )

        pars = Parameters(
            names,
            Array(
                name,
                array(value, dtype=dtype),
                label=label,
                mode="store_weak",
            ),
            fixed=fixed,
            variable=variable,
            close=not has_constraint,
        )

        if has_constraint:
            pars.set_constraint(
                GaussianConstraint.from_numbers(parameters=pars, dtype=dtype, label=label, **kwargs)
            )
            pars._close()

        return pars


class GaussianConstraint(Constraint):
    __slots__ = (
        "central",
        "sigma",
        "normvalue",
        "sigma_total",
        "normvalue_final",
        "_central_node",
        "_sigma_node",
        "_normvalue_node",
        "_cholesky_node",
        "_covariance_node",
        "_correlation_node",
        "_sigma_total_node",
        "_norm_node",
        "_is_constrained",
    )
    central: Output
    sigma: Output
    normvalue: Output
    sigma_total: Output

    normvalue_final: Output

    _central_node: Node
    _sigma_node: Node
    _normvalue_node: Node

    _cholesky_node: Optional[Node]
    _covariance_node: Optional[Node]
    _correlation_node: Optional[Node]
    _sigma_total_node: Optional[Node]

    _norm_node: Node

    _is_constrained: bool

    def __init__(
        self,
        central: Node,
        *,
        parameters: Parameters,
        sigma: Optional[Node] = None,
        covariance: Optional[Node] = None,
        correlation: Optional[Node] = None,
        constrained: Optional[bool] = None,
        free: Optional[bool] = None,
        **_,
    ):
        super().__init__(parameters=parameters)
        self._central_node = central

        self._cholesky_node = None
        self._covariance_node = None
        self._correlation_node = None
        self._sigma_total_node = None

        if all(f is not None for f in (constrained, free)):
            raise InitializationError(
                "GaussianConstraint may not be set to constrained and free at the same time"
            )
        if constrained is not None:
            self._is_constrained = constrained
        elif free is not None:
            self._is_constrained = not free
        else:
            self._is_constrained = True

        if sigma is not None and covariance is not None:
            raise InitializationError(
                'GaussianConstraint: got both "sigma" and "covariance" as arguments'
            )
        if correlation is not None and sigma is None:
            raise InitializationError(
                'GaussianConstraint: got "correlation", but no "sigma" as arguments'
            )

        value_node = parameters._value_node
        self._sigma_total_node = sigma
        if sigma is not None:
            self.sigma_total = sigma.outputs[0]
        if correlation is not None:
            self._correlation_node = correlation
            self._covariance_node = CovmatrixFromCormatrix(f"V({value_node.name})")
            self._cholesky_node = Cholesky(f"L({value_node.name})")
            self._sigma_node = self._cholesky_node

            self._sigma_total_node >> self._covariance_node.inputs["sigma"]
            correlation >> self._covariance_node
            self._covariance_node >> self._cholesky_node
        elif sigma is not None:
            self._sigma_node = sigma
        elif covariance is not None:
            self._cholesky_node = Cholesky(f"L({value_node.name})")
            self._sigma_node = self._cholesky_node
            self._covariance_node = covariance

            covariance >> self._cholesky_node

            # Todo, add square root of the diagonal of the covariance matrix
            self.sigma_total = None
            self._sigma_total_node = None
        else:
            # TODO: no sigma/covariance AND central means normalized=value?
            raise InitializationError(
                'GaussianConstraint: got no "sigma" and no "covariance" arguments'
            )

        self.central = self._central_node.outputs[0]
        self.sigma = self._sigma_node.outputs[0]

        if (mark := value_node.labels.mark) is not None:
            normmark = f"norm({mark})"
        else:
            normmark = "norm"
        self._normvalue_node = Array(
            f"[norm] {value_node.name}",
            zeros_like(self.central._data),
            mark=normmark,
            mode="store_weak",
        )
        self._normvalue_node.labels.inherit(
            self._pars._value_node.labels, fmtlong="[norm] {}", fmtshort="n({})"
        )
        self.normvalue = self._normvalue_node.outputs[0]

        self._norm_node = NormalizeCorrelatedVars2(f"[norm] {value_node.name}", immediate=True)
        self.central >> self._norm_node.inputs["central"]
        self.sigma >> self._norm_node.inputs["matrix"]

        for nodename in ("_cholesky_node", "_covariance_node", "_norm_node"):
            if cnode := getattr(self, nodename):
                cnode.labels.inherit(self._pars._value_node.labels, fields=("index_values",))

        (parameters.value, self.normvalue) >> self._norm_node
        self.normvalue_final = self._norm_node.outputs["normvalue"]

        self._norm_node.close(recursive=True)
        self._norm_node.touch()

        value_output = self._pars.value
        self._pars._reset_pars()
        for i in range(value_output._data.size):
            self._pars._pars.append(
                GaussianParameter(
                    value_output,
                    self.central,
                    self.sigma_total,
                    i,
                    normvalue_output=self.normvalue,
                    connectible=self._norm_node.outputs[0],
                    parent=parameters,
                )
            )
            self._pars._norm_pars.append(
                NormalizedGaussianParameter(self.normvalue, i, parent=parameters, labelfmt="{}")
            )

    @property
    def is_constrained(self) -> bool:
        return self._is_constrained

    @property
    def is_free(self) -> bool:
        return not self._is_constrained

    @property
    def is_correlated(self) -> bool:
        return self._covariance_node is not None

    @staticmethod
    def from_numbers(
        *,
        central: Union[float, Sequence[float]],
        sigma: Union[float, Sequence[float]],
        label: Optional[Dict[str, str]] = None,
        dtype: DTypeLike = "d",
        **kwargs,
    ) -> Parameters:
        if label is None:
            label = {"text": "gaussian parameter"}
        else:
            label = dict(label)
        name = label.setdefault("name", "parameter")

        if isinstance(central, (float, int)):
            central = (central,)
        if isinstance(sigma, (float, int)):
            sigma = (sigma,)

        node_central = Array(
            f"{name}_central",
            array(central, dtype=dtype),
            label=inherit_labels(label, fmtlong="central: {}", fmtshort="c({})"),
            mode="store_weak",
        )

        node_sigma = Array(
            f"{name}_sigma",
            array(sigma, dtype=dtype),
            label=inherit_labels(label, fmtlong="sigma: {}", fmtshort="σ({})"),
            mode="store_weak",
        )

        return GaussianConstraint(central=node_central, sigma=node_sigma, **kwargs)


def GaussianParameters(names: Tuple[Tuple[str]], value: Node, *args, **kwargs) -> Parameters:
    pars = Parameters(names, value, close=False)
    pars.set_constraint(GaussianConstraint(*args, parameters=pars, **kwargs))
    pars._close()

    return pars
