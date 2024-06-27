from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from numba import njit

from ..exception import InitializationError
from ..parameters import GaussianParameter
from .OneToOneNode import OneToOneNode

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..input import Input


class Jacobian(OneToOneNode):
    __slots__ = ("_scale", "_parameters_list")

    _scale: float
    _parameters_list: list[GaussianParameter]

    def __init__(
        self,
        name,
        scale: float = 0.5,
        parameters: Sequence[GaussianParameter] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(name, **kwargs)
        self._scale = scale

        self._parameters_list = []  # pyright: ignore
        if parameters:
            if not isinstance(parameters, Sequence):
                raise InitializationError(
                    f"parameters must be a sequence of GaussianParameter, but given {parameters=},"
                    f" {type(parameters)=}!"
                )
            for par in parameters:
                self.append_par(par)

    def append_par(self, par: GaussianParameter) -> None:
        if not isinstance(par, GaussianParameter):
            raise RuntimeError(f"par must be a GaussianParameter, but given {par=}, {type(par)=}!")
        self._parameters_list.append(par)

    def _typefunc(self) -> None:
        n = len(self._parameters_list)
        for inp, out in zip(self.inputs, self.outputs):
            out.dd.dtype = inp.dd.dtype
            out.dd.shape = (inp.dd.size, n)

    def _fcn(self):
        c1 = 4.0 / 3.0
        c2 = 1.0 / 6.0
        for inp, outdata in zip(self.inputs, self.outputs.iter_data()):
            for i, parameter in enumerate(self._parameters_list):
                reldelta = parameter.sigma * self._scale
                f1 = c1 / reldelta
                f2 = c2 / reldelta

                x0 = parameter.value
                self._do_step(i, parameter, x0 + 0.5 * reldelta, f1, inp, outdata)
                self._do_step(i, parameter, x0 - 0.5 * reldelta, -f1, inp, outdata)
                self._do_step(i, parameter, x0 + reldelta, -f2, inp, outdata)
                self._do_step(i, parameter, x0 - reldelta, f2, inp, outdata)
                parameter.value = x0
                inp.touch()

    def _do_step(
        self,
        icol: int,
        param: GaussianParameter,
        newval: float,
        coeff: float,
        inp: Input,
        res: NDArray,
    ):
        param.value = newval
        inp.touch()
        _step_in_numba(res, inp.data, coeff, icol)


@njit(cache=True)
def _step_in_numba(res: NDArray, inpdata: NDArray, coeff: float, icol: int) -> None:
    for j in range(len(inpdata)):
        res[j, icol] += coeff * inpdata[j]
