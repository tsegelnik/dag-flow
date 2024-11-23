class Node:
    __slots__ = (
        "inputs",
    )

    inputs: list["Node"]

    def __init__(
            self,
            inputs: list["Node"] | None = None,

    ):
        if inputs is not None:
            self.inputs = inputs
        else:
            self.inputs = []

    def run(self):
        return self.fcn()

    def __rshift__(self, other):
        other.inputs.append(self)
        return other

    def fcn(self):
        for inp in self.inputs:
            inp.run()

        return self.inputs[0]