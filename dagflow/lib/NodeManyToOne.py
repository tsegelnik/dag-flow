from ..input_extra import MissingInputAddOne
from ..nodes import FunctionNode
from ..typefunctions import (
    AllPositionals,
    check_has_inputs,
    check_inputs_equivalence,
    copy_input_edges_to_output,
    copy_input_shape_to_output,
    eval_output_dtype,
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
        check_has_inputs(self) # at least one input
        check_inputs_equivalence(self) # all the inputs are have same dd fields
        copy_input_shape_to_output(self, 0, "result") # copy shape to result
        copy_input_edges_to_output(self, 0, "result") # copy edges to result
        eval_output_dtype(self, AllPositionals, "result") # eval dtype of result
