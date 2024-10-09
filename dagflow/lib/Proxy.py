from .ManyToOneNode import ManyToOneNode
from numpy import copyto


class Proxy(ManyToOneNode):
    """Proxy inputs"""
    __slots__ = ("_idx",)
    _idx: int

    def __init__(self, *args, **kwargs):
        self._idx = 0
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "proxy")

    def _fcn(self):
        print(self._input_nodes_callbacks)
        self._input_nodes_callbacks[self._idx]()
        copyto(self._output_data, self._input_data[self._idx])

    def switch_input(self, idx: int) -> None:
        self._idx = idx
        self._post_allocate()
        self.taint()
