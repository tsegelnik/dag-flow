from numpy import array

from ..nodes import StaticNode
from numpy.typing import ArrayLike

class Array(StaticNode):
    """Creates a node with a single data output with predefined array"""

    _data: ArrayLike

    def __init__(self, name, arr, outname="array", **kwargs):
        super().__init__(name, **kwargs)
        output = self._add_output(
            outname, allocatable=False, data=array(arr, copy=True), owns_data=True
        )
        self._data = output._data

    def _fcn(self):
        return self._data

    def _typefunc(self) -> None:
        pass
