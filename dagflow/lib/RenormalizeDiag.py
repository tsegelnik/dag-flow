from typing import TYPE_CHECKING, Callable, Literal

from numba import njit
from numpy.typing import NDArray

from ..inputhandler import MissingInputAddPair
from ..typefunctions import (
    AllPositionals,
    check_input_shape,
    check_input_square,
    check_inputs_equivalence,
)
from .OneToOneNode import OneToOneNode

if TYPE_CHECKING:
    from ..input import Input


class RenormalizeDiag(OneToOneNode):
    __slots__ = ("_mode", "_ndiag", "_scale")

    _mode: str
    _ndiag: int
    _scale: "Input"

    def __init__(
        self, *args, mode: Literal["diag", "offdiag"] = "diag", ndiag: int = 1, **kwargs
    ) -> None:
        super().__init__(
            *args,
            missing_input_handler=MissingInputAddPair(input_fmt="matrix", output_fmt="result"),
            **kwargs,
        )

        if mode in {"diag", "offdiag"}:
            self._labels.setdefault("mark", f"renormalize {mode}")
        else:
            raise RuntimeError(f"Invalid RenormalizeDiag mode {mode}")
        self._mode = mode

        if ndiag < 1:
            raise RuntimeError(f"Invalid RenormalizeDiag ndiag={ndiag} (<1)")
        self._ndiag = ndiag

        self._scale = self._add_input("scale", positional=False)
        self._functions.update({"diag": self._fcn_diag, "offdiag": self._fcn_offdiag})

    def _fcn_diag(self) -> None:
        scale = self._scale.data[0]
        for input, output in zip(self.inputs, self.outputs):
            _renorm_diag_numba(input.data, output._data, scale, self._ndiag)

    def _fcn_offdiag(self) -> None:
        scale = self._scale.data[0]
        for input, output in zip(self.inputs, self.outputs):
            _renorm_offdiag_numba(input.data, output._data, scale, self._ndiag)

    def _typefunc(self) -> None:
        super()._typefunc()
        check_input_shape(self, "scale", (1,))
        check_input_square(self, 0)
        check_inputs_equivalence(self, AllPositionals, check_dtype=True, check_shape=True)
        self.fcn = self._functions[self._mode]


@njit(cache=True)
def zero_to_one(x: float) -> float:
    return x if x != 0.0 else 1.0


@njit(cache=True)
def _norming(matrix: NDArray) -> None:
    for icol in range(matrix.shape[-1]):
        matrix[:, icol] /= zero_to_one(matrix[:, icol].sum())


def _renorm_diag_python(matrix: NDArray, out: NDArray, scale: float, ndiag: float) -> None:
    out[:, :] = matrix[:, :]
    n = out.shape[0]
    # main diag
    for i in range(n):
        out[i, i] *= scale
    # other diagonals
    idiag = 1
    while idiag < ndiag:
        i = 0
        while i + idiag < n:
            out[i + idiag, i] *= scale
            out[i, i + idiag] *= scale
            i += 1
        idiag += 1
    _norming(out)


def _renorm_offdiag_python(matrix: NDArray, out: NDArray, scale: float, ndiag: float) -> None:
    out[:, :] = matrix[:, :] * scale
    n = out.shape[0]
    # main diag
    for i in range(n):
        out[i, i] = matrix[i, i]
    # other diagonals
    idiag = 1
    while idiag < ndiag:
        i = 0
        while i + idiag < n:
            out[i + idiag, i] = matrix[i + idiag, i]
            out[i, i + idiag] = matrix[i, i + idiag]
            i += 1
        idiag += 1
    # norming
    _norming(out)


_renorm_diag_numba: Callable[[NDArray, NDArray, float, float], None] = njit(cache=True)(
    _renorm_diag_python
)
_renorm_offdiag_numba: Callable[[NDArray, NDArray, float, float], None] = njit(cache=True)(
    _renorm_offdiag_python
)
