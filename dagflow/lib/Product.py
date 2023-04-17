from numpy import copyto

from .NodeManyToOne import NodeManyToOne


class Product(NodeManyToOne):
    """Product of all the inputs together"""

    def _fcn(self, _, inputs, outputs):
        out = outputs["result"].data
        copyto(out, inputs[0].data)
        if len(inputs) > 1:
            for input in inputs[1:]:
                out *= input.data
        return out
