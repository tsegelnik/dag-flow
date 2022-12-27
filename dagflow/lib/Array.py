from numpy import array

from ..nodes import FunctionNode
from ..output import Output
from numpy.typing import NDArray
from ..exception import InitializationError

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
        elif mode=='fill':
            self._output = self._add_output(outname, dtype=self._data.dtype, shape=self._data.shape)
        else:
            raise InitializationError(f'Array: invalid mode "{mode}"', node=self)

        self._functions.update({
                "store": self._fcn_store,
                "fill": self._fcn_fill
                })
        self.fcn = self._functions[mode]

    def _fcn_store(self, *args):
        return self._data

    def _fcn_fill(self, *args):
        data = self._output._data
        data[:] = self._data
        return data

    def _typefunc(self) -> None:
        pass
