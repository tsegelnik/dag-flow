from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from numba import njit
from numpy import zeros

from ...core.exception import InitializationError
from ...parameters import AnyGaussianParameter, GaussianParameter, NormalizedGaussianParameter
from ..abstract import OneToOneNode

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ...core.output import Output


class Jacobian(OneToOneNode):
    __slots__ = ("_scale", "_parameters_list")

    _scale: float
    _parameters_list: list[AnyGaussianParameter]

    def __init__(
        self,
        name,
        scale: float = 0.1,
        parameters: Sequence[AnyGaussianParameter] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(name, **kwargs)
        self._scale = scale

        self._parameters_list = []  # pyright: ignore
        if parameters:
            if not isinstance(parameters, Sequence):
                raise InitializationError(
                    f"parameters must be a sequence of GaussianParameter or NormalizedGaussianParameter, but given {parameters=},"
                    f" {type(parameters)=}!"
                )
            for par in parameters:
                self.append_par(par)

    def append_par(self, par: AnyGaussianParameter) -> None:
        if not isinstance(par, (GaussianParameter, NormalizedGaussianParameter)):
            raise RuntimeError(
                f"par must be a GaussianParameter or NormalizedGaussianParameter, but given {par=}, {type(par)=}!"
            )
        self._parameters_list.append(par)

    def _typefunc(self) -> None:
        n = len(self._parameters_list)
        for inp, out in zip(self.inputs, self.outputs):
            out.dd.dtype = inp.dd.dtype
            out.dd.shape = (inp.dd.size, n)

    def _function(self):
        c1 = 4.0 / 3.0
        c2 = 1.0 / 6.0
        for inp, outdata in zip(self.inputs, self.outputs.iter_data_unsafe()):
            outdata[:] = 0.0
            parent_output = inp.parent_output
            for i, parameter in enumerate(self._parameters_list):
                reldelta = parameter.sigma * self._scale
                f1 = c1 / reldelta
                f2 = c2 / reldelta

                x0 = parameter.value
                _do_step(i, parameter, x0 + 0.5 * reldelta, f1, parent_output, outdata)
                _do_step(i, parameter, x0 - 0.5 * reldelta, -f1, parent_output, outdata)
                _do_step(i, parameter, x0 + reldelta, -f2, parent_output, outdata)
                _do_step(i, parameter, x0 - reldelta, f2, parent_output, outdata)
                parameter.value = x0
                inp.touch()
        # We need to set the flag frozen manually
        self.fd.frozen = True

    def compute(self) -> None:
        self.unfreeze()
        self.touch(force_computation=True)


def _do_step(
    icol: int,
    param: AnyGaussianParameter,
    newval: float,
    coeff: float,
    model: Output,
    res: NDArray,
) -> NDArray:
    param.value = newval
    model.touch()
    _step_in_numba(res, model.data, coeff, icol)


def compute_jacobian(
    model: Output,
    parameters: Iterable[AnyGaussianParameter],
    out: NDArray | None = None,
    scale: float = 0.1,
):
    c1 = 4.0 / 3.0
    c2 = 1.0 / 6.0

    if not isinstance(parameters, Sequence):
        parameters = list(parameters)
    npars = len(parameters)

    assert len(model.dd.shape)==1
    if out is None:
        out = zeros((model.dd.shape[0], npars), dtype=model.dd.dtype)

    for i, parameter in enumerate(parameters):
        reldelta = parameter.sigma * scale
        f1 = c1 / reldelta
        f2 = c2 / reldelta

        x0 = parameter.value
        _do_step(i, parameter, x0 + 0.5 * reldelta, f1, model, out)
        _do_step(i, parameter, x0 - 0.5 * reldelta, -f1, model, out)
        _do_step(i, parameter, x0 + reldelta, -f2, model, out)
        _do_step(i, parameter, x0 - reldelta, f2, model, out)
        parameter.value = x0
        model.touch()

    return out


@njit(cache=True)
def _step_in_numba(res: NDArray, inpdata: NDArray, coeff: float, icol: int) -> None:
    for j in range(len(inpdata)):
        res[j, icol] += coeff * inpdata[j]
