from collections.abc import Callable
from typing import Literal
from typing import TYPE_CHECKING

from numba import njit
from numpy.typing import NDArray

from ..inputhandler import MissingInputAddPair
from ..typefunctions import AllPositionals
from ..typefunctions import check_input_shape
from ..typefunctions import check_input_square
from ..typefunctions import check_inputs_equivalence
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

        if ndiag < 1 or not isinstance(ndiag, int):
            raise RuntimeError(
                f"Invalid RenormalizeDiag {ndiag=}, {type(ndiag)=} (must be int >1)!"
            )
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
def _norming(matrix: NDArray) -> None:
    for icol in range(matrix.shape[-1]):
        colsum = matrix[:, icol].sum()
        if colsum != 0.0:
            matrix[:, icol] /= colsum


def _renorm_diag_python(matrix: NDArray, out: NDArray, scale: float, ndiag: int) -> None:
    out[:, :] = matrix[:, :]
    n = out.shape[0]
    # main diagonal
    for i in range(n):
        out[i, i] *= scale
    # other diagonals
    for idiag in range(1, ndiag):
        i = 0
        while i + idiag < n:
            out[i + idiag, i] *= scale
            out[i, i + idiag] *= scale
            i += 1
    _norming(out)


def _renorm_offdiag_python(matrix: NDArray, out: NDArray, scale: float, ndiag: int) -> None:
    out[:, :] = matrix[:, :]
    out *= scale
    n = out.shape[0]
    # main diagonal
    for i in range(n):
        out[i, i] = matrix[i, i]
    # other diagonals
    for idiag in range(1, ndiag):
        i = 0
        while i + idiag < n:
            out[i + idiag, i] = matrix[i + idiag, i]
            out[i, i + idiag] = matrix[i, i + idiag]
            i += 1
    _norming(out)


_renorm_diag_numba: Callable[[NDArray, NDArray, float, float], None] = njit(cache=True)(
    _renorm_diag_python
)
_renorm_offdiag_numba: Callable[[NDArray, NDArray, float, float], None] = njit(cache=True)(
    _renorm_offdiag_python
)
