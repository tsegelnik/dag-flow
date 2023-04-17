from numpy import copyto

from .NodeManyToOne import NodeManyToOne

class Division(NodeManyToOne):
    """Division of all the inputs together"""

    def _fcn(self, _, inputs, outputs):
        out = outputs[0].data
        copyto(out, inputs[0].data.copy())
        if len(inputs) > 1:
            for input in inputs[1:]:
                out /= input.data
        return out
