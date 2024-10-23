class Node:
    __slots__ = (
        "name",
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

    def __call__(self):
        return self.fcn()

    def __rshift__(self, other):
        other.inputs.append(self)
        return other

    def fcn(self):
        pass
