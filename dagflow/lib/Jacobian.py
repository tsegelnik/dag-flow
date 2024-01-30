from typing import TYPE_CHECKING
from collections.abc import Sequence

from numpy.typing import NDArray

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
    _parameters_list: list[Parameter]

    def __init__(
        self,
        name,
        reldelta: float = 0.1,
        step: float = 0.1,
        parameters: Sequence[Parameter] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(name, **kwargs, allowed_kw_inputs=("func",))
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
        # TODO: do we need to check smth?

    def _fcn(self):
        reldelta_corrected = self._reldelta * self._step
        f1 = 4.0 / (3.0 * reldelta_corrected)
        f2 = 1.0 / (6.0 * reldelta_corrected)
        res = self._jacobian.data
        for i, parameter in enumerate(self._parameters_list):
            # TODO: check coefficients
            x0 = parameter.value
            self._do_step(i, parameter, res, self._func, reldelta_corrected / 2.0, f1)
            self._do_step(i, parameter, res, self._func, -reldelta_corrected / 2.0, -f1)
            self._do_step(i, parameter, res, self._func, reldelta_corrected, -f2)
            self._do_step(i, parameter, res, self._func, -reldelta_corrected, f2)
            parameter.value = x0

    def _do_step(
        self, i: int, param: Parameter, res: NDArray, func: "Input", diff: float, coeff: float
    ):
        param.value += diff
        # TODO: do we need to touch, actually?
        # func.touch()
        res[:, i] += coeff * func.data[:]
