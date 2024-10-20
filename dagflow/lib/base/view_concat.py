from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import zeros

from ...node import Node
from ...typefunctions import check_input_dimension, check_input_dtype

if TYPE_CHECKING:
    from ...input import Input
    from ...output import Output


class ViewConcat(Node):
    """Creates a node with a single data output which is a concatenated memory of the inputs"""

    __slots__ = ("_output", "_offsets")
    _output: Output
    _offsets: list[int]

    def __init__(self, name, outname="concat", **kwargs) -> None:
        super().__init__(name, **kwargs)
        self._output = self._add_output(outname, allocatable=False, forbid_reallocation=True)
        self._offsets = []

    def missing_input_handler(self, idx: int | None = None, scope: int | None = None) -> Input:
        idx = idx if idx is not None else len(self.inputs)
        iname = f"input_{idx:02d}"
        return self._add_input(iname, allocatable=True, child_output=self._output)

    def _fcn(self):
        self.inputs.touch()

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        size = 0
        self._offsets = []
        cdtype = self.inputs[0].dd.dtype
        check_input_dtype(self, slice(None), cdtype)
        check_input_dimension(self, slice(None), 1)
        for _input in self.inputs:
            self._offsets.append(size)
            size += _input.dd.shape[0]

        output = self.outputs[0]
        output.dd.dtype = cdtype
        output.dd.shape = (size,)
        data = zeros(shape=size, dtype=cdtype)
        output._set_data(data, owns_buffer=True)

        for offset, _input in zip(self._offsets, self.inputs):
            size = _input.dd.shape[0]
            idata = data[offset : offset + size]
            _input.set_own_data(idata, owns_buffer=False)
