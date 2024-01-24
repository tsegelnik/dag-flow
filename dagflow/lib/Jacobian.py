from typing import TYPE_CHECKING, List, Optional, Sequence

from ..exception import InitializationError
from ..nodes import FunctionNode
from ..parameters import Parameter, Parameters

if TYPE_CHECKING:
    from ..input import Input
    from ..output import Output


class Jacobian(FunctionNode):
    __slots__ = ("_func", "_jacobian", "_reldelta", "_step", "_parameters_list")

    _func: "Input"
    _jacobian: "Output"
    _reldelta: float
    _step: float
    _parameters_list: List[Parameter]

    def __init__(
        self,
        name,
        reldelta: float = 1e-1,
        step: float = 0.1,
        parameters: Optional[Sequence[Parameter]] = None,
        **kwargs,
    ) -> None:
        super().__init__(name, **kwargs)
        # TODO: do we need a check of step and reldelta values?
        self._reldelta = reldelta
        self._step = step

        self._func = self._add_input("func")
        self._jacobian = self._add_output("jacobian")

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

    def append_par(self, par: Parameter) -> None:
        if not isinstance(par, Parameter):
            raise RuntimeError(f"par must be a Parameter, but given {par=}, {type(par)=}!")
        self._parameters_list.append(par)

    def _typefunc(self) -> None:
        self._jacobian.dd.dtype = "d"
        self._jacobian.dd.shape = (self._func.dd.size, len(self._parameters_list))

    def _fcn(self):
        # TODO: implement an algorithm
        pass
