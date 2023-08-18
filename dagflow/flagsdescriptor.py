from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .output import Outputs
    from .input import Inputs


class FlagsDescriptor:
    """
    The class stores the node flags
    """

    __slots__ = (
        "tainted",
        "frozen",
        "frozen_tainted",
        "invalid",
        "closed",
        "allocated",
        "being_evaluated",
        "types_tainted",
        "_children",
        "_parents",
    )

    # flags
    tainted: bool
    frozen: bool
    frozen_tainted: bool
    invalid: bool
    closed: bool
    allocated: bool
    being_evaluated: bool
    types_tainted: bool
    # observers and observed
    _children: "Outputs"
    _parents: "Inputs"

    def __init__(
        self,
        *,
        children: "Outputs",
        parents: "Inputs",
        tainted: bool = True,
        frozen: bool = False,
        frozen_tainted: bool = False,
        invalid: bool = False,
        closed: bool = False,
        allocated: bool = False,
        being_evaluated: bool = False,
        types_tainted: bool = True,
    ) -> None:
        self._children = children
        self._parents = parents
        self.tainted = tainted
        self.frozen = frozen
        self.frozen_tainted = frozen_tainted
        self.invalid = invalid
        self.closed = closed
        self.allocated = allocated
        self.being_evaluated = being_evaluated
        self.types_tainted = types_tainted

    def __str__(self) -> str:
        return ", ".join(f"{slot}={getattr(self, slot)}" for slot in self.__slots__)

    @property
    def children(self) -> "Outputs":
        return self._children

    @property
    def parents(self) -> "Inputs":
        return self._parents

    def _invalidate(self, invalid) -> None:
        if invalid:
            self.invalidate()
        elif any(parent.invalid for parent in self.parents.iter_all()):
            return
        else:
            self.invalidate(False)
        self.invalidate_children(invalid)

    def invalidate(self, invalid: bool = True) -> None:
        self.invalid = invalid
        self.frozen_tainted = False
        self.frozen = False
        self.tainted = True

    def invalidate_children(self, invalid: bool = True) -> None:
        for child in self.children:
            child.invalid = invalid

    def invalidate_parents(self, invalid: bool = True) -> None:
        for parent in self.parents.iter_all():
            node = parent.parent_node
            node.invalidate(invalid)
            node.invalidate_parents(invalid)

    def freeze(self) -> None:
        self.frozen = True
        self.frozen_tainted = False

    def taint_children(self, **kwargs) -> None:
        for child in self.children:
            child.taint_children(**kwargs)

    def taint_type(self, force: bool = False):
        self.types_tainted = True
        self.tainted = True
        self.frozen = False
        for child in self.children:
            child.taint_children_type(force)
