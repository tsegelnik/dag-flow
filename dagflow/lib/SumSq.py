from numpy import add, empty_like, ndarray, square

from .ManyToOneNode import ManyToOneNode


class SumSq(ManyToOneNode):
    """Sum of the squares of all the inputs"""

    _buffer: ndarray

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "Σ()²")

    def _fcn(self):
        out = self.outputs["result"].data
        square(self.inputs[0].data, out=out)
        if len(self.inputs) > 1:
            for _input in self.inputs[1:]:
                square(_input.data, out=self._buffer)
                add(self._buffer, out, out=out)
        return out

    def _post_allocate(self) -> None:
        self._buffer = empty_like(self.inputs[0].data_unsafe)
