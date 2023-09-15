from typing import TYPE_CHECKING, Literal, Optional

from ..exception import InitializationError
from ..nodes import FunctionNode
from ..typefunctions import check_inputs_number, copy_from_input_to_output

if TYPE_CHECKING:
    from ..input import Input
    from ..output import Output


class SegmentIndex(FunctionNode):
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

    _coarse: "Input"
    _fine: "Input"
    _indices: "Output"

    def __init__(
        self,
        *args,
        mode: Literal["left", "right"] = "right",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
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

    def _typefunc(self) -> None:
        """
        The function to determine the dtype and shape of the ouput.
        """
        check_inputs_number(self, 2)
        copy_from_input_to_output(
            self, 1, 0, dtype=False, shape=True, edges=False, meshes=False
        )
        self._indices.dd.dtype = "i"

    def _fcn(self) -> Optional[list]:
        """Uses `numpy.ndarray.searchsorted` and `numpy.ndarray.argsort`"""
        out = self._indices.data.ravel()
        coarse = self._coarse.data.ravel()
        fine = self._fine.data.ravel()
        # NOTE: `searchsorted` and `argsort` allocate a memory!
        #       it is better to use another algorithm if possible
        sorter = coarse.argsort()
        out[:] = coarse.searchsorted(fine, side=self.mode, sorter=sorter)
