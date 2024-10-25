from ..abstract import OneToOneNode


class Copy(OneToOneNode):
    """Copy/identity function"""
    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "copy")

    def _function(self):
        for inp, out in zip(self.inputs, self.outputs):
            out.data[:] = inp.data
