from numba import njit
from numpy import integer
from numpy.typing import NDArray

from ..exception import TypeFunctionError
from ..typefunctions import (
    AllPositionals,
    check_has_inputs,
    check_input_dimension,
    check_input_shape,
    check_input_subtype,
    copy_input_dtype_to_output,
)
from .OneToOneNode import OneToOneNode


@njit(cache=True)
def _psum(data: NDArray, range: NDArray, out: NDArray):
    out[0] = data[range[0] : range[1]].sum()


class PartialSums(OneToOneNode):
    """
    inputs:
        `a`: array to sum
        `0`, `...`: range to partial sum

    outputs:
        `0`, `...`: partial sum

    The node performs partial sums of the input array `a`
    within ranges which are the positional inputs.

    .. note:: now works only with 1d arrays
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._add_input("a", positional=False)

    def _typefunc(self) -> None:
        check_has_inputs(self, "a")
        check_has_inputs(self, AllPositionals)
        check_input_dimension(self, (AllPositionals, "a"), 1)
        check_input_subtype(self, AllPositionals, integer)
        check_input_shape(self, AllPositionals, (2,))
        copy_input_dtype_to_output(self, "a", AllPositionals)
        for out in self.outputs:
            out.dd.shape = (1,)
        # TODO: axes_edges and axes_meshes?
        # now edges are restricted
        a = self.inputs["a"]
        if a.dd.axes_edges:
            raise TypeFunctionError(
                "The PartialSums doesn't support edges functional, "
                f"but given {a.dd.axes_edges}",
                node=self,
                input=a,
            )

    def _fcn(self, _, inputs, outputs):
        data = inputs["a"].data
        for inp, out in zip(inputs, outputs):
            _psum(data, inp.data, out.data)
        return list(outputs.iter_data())
