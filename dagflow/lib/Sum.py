from numpy import add, copyto

from .NodeManyToOne import NodeManyToOne


class Sum(NodeManyToOne):
    """Sum of all the inputs together"""

    def _fcn(self, _, inputs, outputs):
        out = outputs["result"].data
        copyto(out, inputs[0].data)
        if len(inputs) > 1:
            for input in inputs[1:]:
                add(out, input.data, out=out)
        return out
