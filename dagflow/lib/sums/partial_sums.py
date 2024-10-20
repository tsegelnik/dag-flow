from __future__ import annotations

from typing import TYPE_CHECKING

from numba import njit
from numpy import integer

from ...exception import TypeFunctionError
from ...typefunctions import (
    AllPositionals,
    check_has_inputs,
    check_input_dimension,
    check_input_shape,
    check_input_subtype,
    copy_input_dtype_to_output,
)
from ..abstract import OneToOneNode

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ...input import Input


@njit(cache=True)
def _psum(data: NDArray, range: NDArray, out: NDArray):
    out[0] = data[range[0] : range[1]].sum()


class PartialSums(OneToOneNode):
    """
    inputs:
        `array`: array to sum
        `0`, `...`: range to partial sum

    outputs:
        `0`, `...`: partial sum

    The node performs partial sums of the input array `a`
    within ranges which are the positional inputs.

    .. note:: now works only with 1d arrays
    """

    __slots__ = ("_array",)
    _array: Input

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._array = self._add_input("array", positional=False)

    def _typefunc(self) -> None:
        check_has_inputs(self, "array")
        check_has_inputs(self, AllPositionals)
        check_input_dimension(self, (AllPositionals, "array"), 1)
        check_input_subtype(self, AllPositionals, integer)
        check_input_shape(self, AllPositionals, (2,))
        copy_input_dtype_to_output(self, "array", AllPositionals)
        for out in self.outputs:
            out.dd.shape = (1,)
        # TODO: axes_edges and axes_meshes?
        # now edges are restricted
        add = self._array.dd
        if add.axes_edges:
            raise TypeFunctionError(
                "The PartialSums doesn't support edges functional, " f"but given {add.axes_edges}",
                node=self,
                input=self._array,
            )

    def _fcn(self):
        data = self._array.data
        for inp, out in zip(self.inputs, self.outputs):
            _psum(data, inp.data, out.data)
