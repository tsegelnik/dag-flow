from numpy import array

from ..nodes import FunctionNode
from ..output import Output
from numpy.typing import NDArray
from ..exception import InitializationError
from numpy.typing import ArrayLike

class Array(FunctionNode):
    """Creates a node with a single data output with predefined array"""

    _mode: str
    _data: NDArray
    _output = Output
    def __init__(self, name, arr, outname="array", mode="store", **kwargs):
        super().__init__(name, **kwargs)
        self._mode = mode
        self._data = array(arr, copy=True)

        if mode=='store':
            self._output = self._add_output(outname, data=self._data)
        elif mode=='store_weak':
            self._output = self._add_output(outname, data=self._data, owns_buffer=False)
        elif mode=='fill':
            self._output = self._add_output(outname, dtype=self._data.dtype, shape=self._data.shape)
        else:
            raise InitializationError(f'Array: invalid mode "{mode}"', node=self)

        self._functions.update({
                "store": self._fcn_store,
                "store_weak": self._fcn_store,
                "fill": self._fcn_fill
                })
        self.fcn = self._functions[self._mode]

    def _fcn_store(self, *args):
        return self._data

    def _fcn_fill(self, *args):
        data = self._output._data
        data[:] = self._data
        return data

    def _typefunc(self) -> None:
        pass

    def _post_allocate(self) -> None:
        if self._mode=='fill':
            return

        self._data = self._output._data

    def set(self, data: ArrayLike, check_taint: bool=False) -> bool:
        return self._output.set(data, check_taint)
