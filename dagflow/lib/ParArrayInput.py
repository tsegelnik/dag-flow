from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..exception import InitializationError, TypeFunctionError
from ..node import Node
from ..parameters import Parameter, Parameters
from ..typefunctions import check_input_dimension, check_inputs_number

if TYPE_CHECKING:
    from ..input import Input


class ParArrayInput(Node):
    """Set values for parameters list from an input"""

    __slots__ = ("_parameters_list", "_values")

    _parameters_list: list[Parameter]
    _values: Input

    def __init__(
        self, name, parameters: Sequence[Parameter] | Parameters | None = None, **kwargs
    ) -> None:
        super().__init__(name, **kwargs)
        self._parameters_list = []  # pyright: ignore
        if parameters:
            if isinstance(parameters, Parameters):
                for par in parameters._pars:
                    self.append_par(par)
            elif isinstance(parameters, Sequence):
                for par in parameters:
                    self.append_par(par)
            else:
                raise InitializationError(
                    f"parameters must be a sequence of Parameters, but given {parameters=},"
                    f" {type(parameters)=}!"
                )
        self._values = self._add_input("values")

    def append_par(self, par: Parameter) -> None:
        if not isinstance(par, Parameter):
            raise RuntimeError(f"par must be a Parameter, but given {par=}, {type(par)=}!")
        self._parameters_list.append(par)

    def _typefunc(self) -> None:
        # TODO: check dtype of input and parameters?
        check_inputs_number(self, 1)
        check_input_dimension(self, 0, 1)
        if (npar := len(self._parameters_list)) != (inpsize := self._values.dd.size):
            raise TypeFunctionError(
                (
                    f"The number of parameters ({npar}) must coincide with the input length"
                    f" ({inpsize})!"
                ),
                node=self,
            )

    def _fcn(self) -> None:
        for par, val in zip(self._parameters_list, self._values.data):
            par.value = val
