from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from numba import njit

from ...core.exception import InitializationError, CalculationError
from ...core.node import Node
from ...core.type_functions import check_number_of_inputs, copy_from_inputs_to_outputs

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ...core.input import Input
    from ...core.output import Output


@njit(cache=True)
def _is_sorted(array: NDArray) -> bool:
    previous = array[0]
    for i in range(1, len(array)):
        current = array[i]

        if current <= previous:
            return False

        previous = current

    return True


class SegmentIndex(Node):
    """
    inputs:
        `0` or `coarse`: array of the coarse x points
        `1` or  `fine`: array of the fine x points

    outputs:
        `0` or `indices`: array of the indices of the coarse segments for every fine point

    The node finds an index of the segment in the coarse array for every fine point.
    The node uses `numpy.searchsorted` method. There is an extra argument `mode`:
        `left`: `a[i-1] < v <= a[i]`
        `right`: `a[i-1] <= v < a[i]`
    """

    __slots__ = (
        "_mode",
        "_coarse",
        "_fine",
        "_indices",
    )

    _coarse: Input
    _fine: Input
    _indices: Output

    def __init__(
        self,
        *args,
        mode: Literal["left", "right"] = "right",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs, allowed_kw_inputs=("coarse", "fine"))
        self._labels.setdefault("mark", "[i]")
        if mode not in {"left", "right"}:
            raise InitializationError(
                f"Argument `mode` must be 'left' or 'right', but given '{mode}'!",
                node=self,
            )
        self._mode = mode
        self._coarse = self._add_input("coarse")  # 0
        self._fine = self._add_input("fine")  # 1
        self._indices = self._add_output("indices")  # 0

    @property
    def mode(self) -> str:
        return self._mode

    def _type_function(self) -> None:
        """
        The function to determine the dtype and shape of the ouput.
        """
        check_number_of_inputs(self, 2)
        copy_from_inputs_to_outputs(self, 1, 0, dtype=False, shape=True, edges=False, meshes=False)
        self._indices.dd.dtype = "i"

    def _function(self):
        """Uses `numpy.ndarray.searchsorted` and `numpy.ndarray.argsort`"""
        out = self._indices._data.ravel()
        coarse = self._coarse.data.ravel()
        fine = self._fine.data.ravel()
        if not _is_sorted(coarse):
            raise CalculationError("Coarse array is not sorted", node=self, input=self._coarse)
        out[:] = coarse.searchsorted(fine, side=self.mode)
