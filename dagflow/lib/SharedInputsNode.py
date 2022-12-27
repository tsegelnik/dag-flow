from typing import Optional
from numpy.typing import NDArray
from numpy import zeros

from ..nodes import FunctionNode
from ..exception import TypeFunctionError


class SharedInputsNode(FunctionNode):
    """Creates a node with the same shared data array allocated on the inputs"""

    _data: NDArray

    def __init__(self, name: str, outname: str = "output", **kwargs):
        super().__init__(name, **kwargs)
        self._add_output(outname, allocatable=False)

    def missing_input_handler(
        self, idx: Optional[int] = None, scope: Optional[int] = None
    ):
        icount = len(self.inputs)
        idx = idx if idx is not None else icount
        iname = "input_{:02d}".format(idx)

        kwargs = {"child_output": self.outputs[0]}
        return (
            self._add_input(iname, allocatable=False, **kwargs)
            if icount > 0
            else self._add_input(iname, allocatable=True, **kwargs)
        )

    def _fcn(self, _, inputs, outputs) -> None:
        self.inputs.touch()

    def _typefunc(self) -> None:
        dtype, shape = None, None

        for input in self.inputs[:1]:
            dtype, shape = input.dtype, input.shape

        if dtype is None or shape is None:
            raise TypeFunctionError("Input data is undefined", node=self)

        for input in self.inputs[1:]:
            if input.dtype != dtype or input.shape != shape:
                raise TypeFunctionError(
                    f"Input data is inconsistent with {dtype} [{shape}]",
                    node=self,
                    input=input,
                )

        self._data = zeros(shape=shape, dtype=dtype)
        for input in self.inputs:
            input.own_data = self._data

        self.outputs[0]._set_data(self._data, owns_data=True)
