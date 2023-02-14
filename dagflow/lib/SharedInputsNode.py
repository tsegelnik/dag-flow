from typing import Optional, List

from numpy import zeros
from numpy.typing import NDArray

from ..nodes import FunctionNode, Node
from ..typefunctions import check_has_input, check_inputs_equivalence


class SharedInputsNode(FunctionNode):
    """Creates a node with the same shared data array allocated on the inputs"""

    _data: NDArray
    _parent_nodes: List[Node]

    def __init__(self, name: str, outname: str = "output", **kwargs):
        super().__init__(name, **kwargs)
        self._add_output(outname, allocatable=False, forbid_reallocation=True)

    def missing_input_handler(
        self, idx: Optional[int] = None, scope: Optional[int] = None
    ):
        icount = len(self.inputs)
        idx = idx if idx is not None else icount
        iname = "input_{:02d}".format(idx)

        kwargs = {
            "child_output": self.outputs[0],
            "allocatable": True,
        }
        return self._add_input(iname, **kwargs)

    def _fcn(self, _, inputs, outputs) -> None:
        self.inputs.touch()

    def _typefunc(self) -> None:
        check_has_input(self)
        check_inputs_equivalence(self)

        self._parent_nodes = []

        self._data = zeros(
            shape=self.inputs[0].shape, dtype=self.inputs[0].dtype
        )
        for i, input in enumerate(self.inputs):
            input.set_own_data(self._data, owns_buffer=(i == 0))

            self._parent_nodes.append(input.parent_node)

        self.outputs[0]._set_data(
            self._data, owns_buffer=False, forbid_reallocation=True
        )

    def _on_taint(self, caller: Node) -> None:
        for parent in self._parent_nodes:
            if parent is caller:
                continue
            parent.taint(caller=None)
