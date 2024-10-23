from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

from ..labels import repr_pretty
from ..node import Node
from ..output import Output

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .parameters import Parameters


class Parameter:
    __slots__ = (
        "_idx",
        "_parent",
        "_common_output",
        "_value_output",
        "_view",
        "_common_connectible_output",
        "_connectible_output",
        "_labelfmt",
        "_stack",
    )
    _parent: Parameters
    _idx: int
    _common_output: Output
    _value_output: Output
    _view: Node | None
    _common_connectible_output: Output
    _connectible_output: Output
    _labelfmt: str
    _stack: list[float] | list[int]

    def __init__(
        self,
        value_output: Output,
        idx: int | None = None,
        *,
        parent: Parameters,
        connectible: Output | None = None,
        labelfmt: str = "{}",
        label: Mapping = {},
        make_view: bool = True,
    ):
        self._parent = parent
        self._common_output = value_output
        self._labelfmt = labelfmt

        if connectible is not None:
            self._common_connectible_output = connectible
        else:
            self._common_connectible_output = value_output

        if idx is None:
            self._idx = 0
            self._value_output = self._common_connectible_output
            self._connectible_output = self._common_connectible_output
            self._view = None
        elif make_view:
            self._idx = idx
            label_parent = self._common_output.node.labels.copy()
            if not label:
                label = label_parent
            try:
                idxtuple = parent._names[idx]
                idxname = ".".join(idxtuple)
            except (ValueError, IndexError):
                idxname = "???"
                idxtuple = None

            with suppress(KeyError, IndexError):
                label["paths"] = [label_parent["paths"][idx]]

            from ..lib.common import View  # fmt: skip
            self._view = View(
                f"{self._common_output.node.name}.{idxname}",
                self._common_connectible_output,
                start=idx,
                length=1,
            )
            self._view.labels.inherit(
                label,
                fmtlong=f"{{}} (par {idx}: {idxname})",
                fmtextra={"graph": f"{{source.text}}\\nparameter {idx}: {idxname}"},
                fields_exclude={"paths"},
            )
            # if idxtuple:
            #     self._view.labels.index_values.extend(idxtuple)
            self._value_output = self._view.outputs[0]
            self._connectible_output = self._value_output
        else:
            self._idx = idx
            self._view = None
            self._connectible_output = None
            self._value_output = value_output
        self._stack = []

    def __str__(self) -> str:
        return f"par v={self.value}"

    _repr_pretty_ = repr_pretty

    @property
    def value(self) -> float | int:
        return self._common_output.data[self._idx]

    @value.setter
    def value(self, value: float | int):
        return self._common_output.seti(self._idx, value)

    @property
    def output(self) -> Output:
        return self._value_output

    @property
    def is_correlated(self) -> bool:
        return self._parent.is_correlated

    @property
    def connectible(self) -> Output:
        return self._connectible_output

    def label(self, source: str = "text") -> str:
        return self._labelfmt.format(self._value_output.node.labels[source])

    def to_dict(self, *, label_from: str = "text") -> dict:
        return {"value": self.value, "label": self.label(label_from), "flags": ""}

    def __rshift__(self, other):
        self._connectible_output >> other

    def push(self, other: float | int | None = None) -> float | int:
        self._stack.append(self.value)
        if other is not None:
            if not isinstance(other, (float, int)):
                raise RuntimeError(
                    f"`other` must be float|int|None, but given {other=}, {type(other)=}"
                )
            self.value = other
        return self.value

    def pop(self) -> float | int:
        with suppress(IndexError):
            self.value = self._stack.pop()
        return self.value

    def __enter__(self) -> float | int:
        return self.push()

    def __exit__(self, exc_type, exc_val, exc_tb) -> float | int:
        return self.pop()


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
        idx: int | None = None,
        *,
        normvalue_output: Output,
        **kwargs,
    ):
        super().__init__(value_output, idx, **kwargs)
        self._central_output = central_output
        self._sigma_output = sigma_output
        self._normvalue_output = normvalue_output

    def __str__(self) -> str:
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
        return self._central_output.data[self._idx]

    @central.setter
    def central(self, central: float):
        self._central_output.seti(self._idx, central)

    @property
    def central_output(self) -> Output:
        return self._central_output

    @property
    def sigma(self) -> float:
        return self._sigma_output.data[self._idx]

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
        return self._normvalue_output.data[self._idx]

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
    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, make_view=False, **kwargs)

    def __str__(self) -> str:
        return f"ngpar v={self.value}"

    @property
    def central(self) -> float:
        return 0.0

    @property
    def sigma(self) -> float:
        return 1.0

    @property
    def normvalue(self) -> float:
        return self.value

    @normvalue.setter
    def normvalue(self, normvalue: float):
        self.value = normvalue

    def to_dict(self, **kwargs) -> dict:
        dct = super().to_dict(**kwargs)
        dct.update(
            {
                "central": 0.0,
                "sigma": 1.0,
            }
        )
        return dct


AnyGaussianParameter = GaussianParameter | NormalizedGaussianParameter
