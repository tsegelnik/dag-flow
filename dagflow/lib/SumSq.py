from numpy import add, empty, ndarray, square

from .ManyToOneNode import ManyToOneNode


class SumSq(ManyToOneNode):
    """Sum of the squares of all the inputs"""

    __slots__ = ("_buffer",)
    _buffer: ndarray

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "Σ()²")

    def _fcn(self) -> ndarray:
        out = self.outputs["result"].data
        square(self.inputs[0].data, out=out)
        if len(self.inputs) > 1:
            for _input in self.inputs[1:]:
                square(_input.data, out=self._buffer)
                add(self._buffer, out, out=out)
        return out

    def _post_allocate(self) -> None:
        inpdd = self.inputs[0].dd
        self._buffer = empty(shape=inpdd.shape, dtype=inpdd.dtype)
