from typing import List, Optional, Sequence, TYPE_CHECKING

from ..exception import InitializationError, TypeFunctionError
from ..nodes import FunctionNode
from ..parameters import Parameter

if TYPE_CHECKING:
    from ..input import Input


class ParArrayInput(FunctionNode):
    """Set values for parameters list from an input"""

    __slots__ = ("_parameters_list", "_values")

    _parameters_list: List[Parameter]
    _values: "Input"

    def __init__(self, name, parameters: Optional[List[Parameter]] = None, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self._parameters_list = []  # pyright: ignore
        if parameters:
            if not isinstance(parameters, Sequence):
                raise InitializationError(
                    f"parameters must be a sequence of Parameters, but given {parameters=},"
                    f" {type(parameters)=}!"
                )
            for par in parameters:
                self.append_par(par)
        self._values = self._add_input("values")

    def append_par(self, par: Parameter) -> None:
        if not isinstance(par, Parameter):
            raise RuntimeError(f"par must be a Parameter, but given {par=}, {type(par)=}!")
        self._parameters_list.append(par)

    def _typefunc(self) -> None:
        # TODO: check dtype of input and parameters?
        from ..typefunctions import check_input_dimension, check_inputs_number

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

    def _post_allocate(self) -> None:
        for par, val in zip(self._parameters_list, self._values.data):
            par.value = val
