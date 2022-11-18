from ..nodes import FunctionNode

class View(FunctionNode):
    """Creates a node with a single data output which is a view on the input"""

    def __init__(self, name, outname="view", **kwargs):
        super().__init__(name, **kwargs)
        self._add_input('input')
        output = self._add_output(outname, allocatable=False)

    def _fcn(self, _, inputs, outputs):
        return self.inputs[0].data

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        input = self.inputs[0]
        output = self.outputs[0]
        output._shape = input.shape
        output._dtype = input.dtype
        output.data = input.parent_output._data.view()

        self.logger.debug(
            f"Node '{self.name}': dtype={self.outputs[0].dtype}, "
            f"shape={self.outputs[0].shape}"
        )
