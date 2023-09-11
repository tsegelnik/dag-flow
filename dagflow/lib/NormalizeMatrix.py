from typing import Literal

from numpy import divide, sum
from numpy.typing import NDArray

from ..input_extra import MissingInputAddPair
from ..typefunctions import (
    AllPositionals,
    check_input_dimension,
    check_inputs_equivalence,
)
from .OneToOneNode import OneToOneNode


class NormalizeMatrix(OneToOneNode):
    __slots__ = ("_mode",)

    _mode: str

    def __init__(
        self, *args, mode: Literal["columns", "rows"] = "columns", **kwargs
    ) -> None:
        super().__init__(
            *args,
            missing_input_handler=MissingInputAddPair(
                input_fmt="matrix", output_fmt="result"
            ),
            **kwargs,
        )
        self._mode = mode

        if mode == "columns":
            self._labels.setdefault("mark", "norm cols")
        elif mode == "rows":
            self._labels.setdefault("mark", "norm rows")
        else:
            raise RuntimeError(f"Invalid NormalizeMatrix mode {mode}")

        self._functions.update(
            {
                "columns": self._fcn_norm_columns,
                "rows": self._fcn_norm_rows,
            }
        )

    def _fcn_norm_rows(self) -> None:
        for input, output in zip(self.inputs, self.outputs):
            _norm_rows(input.data, output._data)

    def _fcn_norm_columns(self) -> None:
        for input, output in zip(self.inputs, self.outputs):
            _norm_columns(input.data, output._data)

    def _typefunc(self) -> None:
        super()._typefunc()
        check_input_dimension(self, AllPositionals, ndim=2)
        check_inputs_equivalence(
            self, AllPositionals, check_dtype=True, check_shape=True
        )

        self.fcn = self._functions[self._mode]

    # def _post_allocate(self) -> None:
    #     out = self.outputs[0]
    #     if self._mode == "columns":
    #         self._buf = empty_like(out._data[0])
    #     else:
    #         self._buf = empty_like(out._data[:, 0])


from numba import njit


@njit(cache=True)
def _norm_rows(matrix: NDArray, out: NDArray):
    ncols = matrix.shape[1]
    for row in range(matrix.shape[0]):
        sum = 0.0
        for column in range(ncols):
            sum += matrix[row, column]
        if sum == 0.0:
            out[row, :] = 0.0
            continue
        for column in range(ncols):
            out[row, column] += matrix[row, column] / sum


@njit(cache=True)
def _norm_columns(matrix: NDArray, out: NDArray):
    nrows = matrix.shape[0]
    for column in range(matrix.shape[1]):
        sum = 0.0
        for row in range(nrows):
            sum += matrix[row, column]
        if sum == 0.0:
            out[:, column] = 0.0
            continue
        for row in range(nrows):
            out[row, column] += matrix[row, column] / sum


def _norm_rows_python(matrix: NDArray, out: NDArray, buffer: NDArray):
    sum(matrix, axis=1, out=buffer)
    divide(matrix, buffer[:, None], out=out)


def _norm_columns_python(matrix: NDArray, out: NDArray, buffer: NDArray):
    sum(matrix, axis=0, out=buffer)
    divide(matrix, buffer, out=out)
