from __future__ import annotations

from .OneToOneNode import OneToOneNode


class ArraySum(OneToOneNode):
    """
    inputs:
        `array`: array to sum
        `array_1`: array to sum
        `...`: ...

    outputs:
        `0`, `...`: sum

    The node performs sum of elements of input arrays
    """

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "Σᵢ")

    def _typefunc(self) -> None:
        from ..typefunctions import (
            AllPositionals,
            check_has_inputs,
            copy_input_dtype_to_output,
        )

        check_has_inputs(self, AllPositionals)
        copy_input_dtype_to_output(self, AllPositionals, AllPositionals)
        for out in self.outputs:
            out.dd.shape = (1,)

    def _fcn(self):
        for inp, out in zip(self.inputs, self.outputs):
            out.data[0] = inp.data.sum()
