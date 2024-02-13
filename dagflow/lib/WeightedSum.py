from typing import TYPE_CHECKING

from numpy import copyto

from ..exception import TypeFunctionError
from ..typefunctions import check_has_inputs
from ..typefunctions import copy_input_shape_to_outputs
from ..typefunctions import eval_output_dtype
from .ManyToOneNode import ManyToOneNode

if TYPE_CHECKING:
    from ..input import Input


class WeightedSum(ManyToOneNode):
    """Weighted sum of all the inputs together"""

    __slots__ = ("_weight",)
    _weight: "Input"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, allowed_kw_inputs=("weight",))
        self._weight = self._add_input("weight", positional=False)
        self._functions.update({"number": self._fcn_number, "iterable": self._fcn_iterable})

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        super()._typefunc()
        check_has_inputs(self, "weight")
        weight = self._weight
        shape = weight.dd.shape[0]
        leninp = len(self.inputs)
        if shape == 0:
            raise TypeFunctionError(
                "Cannot use WeightedSum with empty 'weight'!",
                node=self,
                input=weight,
            )
        elif shape == 1:
            self.fcn = self._functions["number"]
        elif shape == leninp:
            self.fcn = self._functions["iterable"]
        else:
            raise TypeFunctionError(
                f"The number of weights (={shape}) must coincide "
                f"with the number of inputs (={leninp})!",
                node=self,
                input=weight,
            )
        copy_input_shape_to_outputs(self, 0, "result")
        eval_output_dtype(self, slice(None), "result")

    def _fcn_number(self):
        """
        The function for one weight for all inputs:
        `len(weight) == 1`
        """
        out = self._result_output.data
        weight = self._weight.data
        copyto(out, self.inputs[0].data.copy())
        for _input in self.inputs[1:]:
            out += _input.data
        out *= weight

    def _fcn_iterable(self):
        """
        The function for one weight for every input:
        `len(weight) == len(inputs)`
        """
        out = self._result_output.data
        weights = self._weight.data
        copyto(out, self.inputs[0].data * weights[0])
        for _input, weight in zip(self.inputs[1:], weights[1:]):
            out += _input.data * weight
