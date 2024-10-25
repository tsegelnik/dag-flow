from ..abstract import OneToOneNode


class Cache(OneToOneNode):
    """Copy/identity/cache function, which freezes after operation"""
    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, auto_freeze=True, **kwargs)
        self._labels.setdefault("mark", "cache")

    def _function(self):
        for inp, out in zip(self.inputs, self.outputs):
            out.data[:] = inp.data

    def recache(self) -> None:
        self.unfreeze()
        self.touch(force_computation=True)
