from ..input_extra import MissingInputAddPair
from ..nodes import FunctionNode
from ..typefunctions import check_has_inputs

class OneToOneNode(FunctionNode):
    """
    The abstract node with an output for every positional input
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("missing_input_handler", MissingInputAddPair())
        super().__init__(*args, **kwargs)

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        from ..typefunctions import (
            check_has_inputs,
            copy_from_input_to_output,
            assign_outputs_axes_from_inputs
        )
        check_has_inputs(self)
        copy_from_input_to_output(self, slice(None), slice(None), edges=True, nodes=True)
        assign_outputs_axes_from_inputs(self, slice(None), slice(None), assign_nodes=True, ignore_assigned=True, ignore_Nd=True)
