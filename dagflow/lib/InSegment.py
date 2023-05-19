from typing import Literal, Optional

from numpy import searchsorted

from ..exception import InitializationError
from ..nodes import FunctionNode
from ..typefunctions import (
    check_if_input_sorted,
    check_input_dimension,
    check_inputs_number,
    copy_from_input_to_output,
)


class InSegment(FunctionNode):
    """
    inputs:
        `0` or `coarse`: array of the coarse x points
        `1` or  `fine`: array of the fine x points

    outputs:
        `0` or `indices`: array of the indices of the segments

    The node finds an index of the segment in the coarse array for every fine point.
    The node uses `numpy.searchsorted` method. There is an extra argument `mode`:
        `left`: `a[i-1] < v <= a[i]`
        `right`: `a[i-1] <= v < a[i]`
    """

    def __init__(
        self,
        *args,
        mode: Literal["left", "right"] = "left",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if mode not in {"left", "right"}:
            raise InitializationError(
                f"Argument `mode` must be 'left' or 'right', but given '{mode}'!",
                node=self,
            )
        self._mode = mode
        self._add_input("coarse")  # 0
        self._add_input("fine")  # 1
        self._add_output("indices")  # 0

    @property
    def mode(self) -> str:
        return self._mode

    def _typefunc(self) -> None:
        """
        The function to determine the dtype and shape of the ouput.
        """
        # NOTE: Now InSegment supports only 1d arrays
        check_inputs_number(self, 2)
        check_input_dimension(self, slice(None), 1)
        check_if_input_sorted(self, 0)
        copy_from_input_to_output(self, 1, 0, False, True, False, False)  # use fine points shape
        self.outputs[0].dd.dtype = "i"

    def _fcn(self, _, inputs, outputs) -> Optional[list]:
        """Uses `numpy.searchsorted`"""
        out = outputs[0].data
        coarse = inputs[0].data
        fine = inputs[1].data
        out[:] = searchsorted(coarse, fine, side=self.mode)
        if self.debug:
            return out
