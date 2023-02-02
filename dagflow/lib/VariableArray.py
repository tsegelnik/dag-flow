from numpy import array

from .Array import Array
from ..output import Output
from numpy.typing import ArrayLike

class VariableArray(Array):
    """Creates a node with a single data output with predefined array, enables editing"""

    def __init__(self, name, arr, *, outname="array", **kwargs):
        super(Array, self).__init__(name, **kwargs)
        self._mode = 'store'
        self._output = self._add_output(
            outname, data=array(arr, copy=True), settable=True
        )
        self._init_fcn()
        self.close()

    def set(self, data: ArrayLike, check_taint: bool=False) -> bool:
        return self._output.set(data, check_taint)
