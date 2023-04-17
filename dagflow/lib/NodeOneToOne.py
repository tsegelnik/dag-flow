from ..input_extra import MissingInputAddEach
from ..nodes import FunctionNode
from ..typefunctions import check_has_inputs


class NodeOneToOne(FunctionNode):
    """
    The abstract node with an output for every positional input
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("missing_input_handler", MissingInputAddEach())
        super().__init__(*args, **kwargs)

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        check_has_inputs(self)
        for inp, out in zip(self.inputs, self.outputs):
            out.dd.axes_edges = inp.dd.axes_edges
            out.dd.axes_nodes = inp.dd.axes_nodes
            out.dd.dtype = inp.dd.dtype
            out.dd.shape = inp.dd.shape
