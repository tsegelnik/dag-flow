from ..input_extra import MissingInputAddOne
from ..nodes import FunctionNode
from ..typefunctions import (
    check_has_inputs,
    eval_output_dtype,
    copy_input_shape_to_output,
    check_inputs_equivalence,
    AllPositionals
)

class NodeManyToOne(FunctionNode):
    """
    The abstract node with only one output `result`,
    which is the result of some function on all the positional inputs
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault(
            "missing_input_handler", MissingInputAddOne(output_fmt="result")
        )
        super().__init__(*args, **kwargs)

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        check_has_inputs(self)
        copy_input_shape_to_output(self, 0, "result")
        check_inputs_equivalence(self)
        eval_output_dtype(self, AllPositionals, "result")
