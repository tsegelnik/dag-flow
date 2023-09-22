from numba import njit
from numpy import double
from numpy.typing import NDArray

from .OneToOneNode import OneToOneNode


@njit(cache=True)
def _bincenter(edges: NDArray[double], centers: NDArray[double]) -> None:
    nbins = len(centers)
    for i in range(nbins):
        centers[i] = (edges[i] + edges[i + 1]) / 2.0


class BinCenter(OneToOneNode):
    """
    The node finds the centers of bins by edges

    inputs:
        `i`: array with bin edges (N)

    outputs:
        `i`: array with centers of bins (N-1)
    """

    def _fcn(self):
        for _input, _output in zip(self.inputs.iter_data(), self.outputs.iter_data()):
            _bincenter(_input, _output)

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        from ..typefunctions import (
            check_has_inputs,
            check_input_dimension,
            AllPositionals,
        )

        check_has_inputs(self, check_named=True)
        check_input_dimension(self, (AllPositionals, *self.inputs.kw.keys()), ndim=1)
        for _input, _output in zip(self.inputs, self.outputs):
            inputdd = _input.dd
            _output.dd.dtype = inputdd.dtype
            _output.dd.shape = (inputdd.shape[0] - 1,)
