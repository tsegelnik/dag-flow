from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .input import Inputs
    from .output import Outputs


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
        "needs_reallocation",
        "needs_postallocate",
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
    needs_reallocation: bool
    needs_postallocate: bool
    being_evaluated: bool
    types_tainted: bool
    # observers and observed
    # _node: Node
    _children: Outputs  # TODO: List[FlagsDescriptor]?
    _parents: Inputs  # TODO: List[FlagsDescriptor]?

    def __init__(
        self,
        *,
        children: Outputs,
        parents: Inputs,
    ) -> None:
        self._children = children
        self._parents = parents
        self.tainted = True
        self.frozen = False
        self.frozen_tainted = False
        self.invalid = False
        self.closed = False
        self.allocated = False
        self.being_evaluated = False
        self.types_tainted = True
        self.needs_reallocation = False
        self.needs_postallocate = False

    def __str__(self) -> str:
        return ", ".join(f"{slot}={getattr(self, slot)}" for slot in self.__slots__)

    @property
    def children(self) -> Outputs:
        return self._children

    @property
    def parents(self) -> Inputs:
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
        if self.types_tainted and not force:
            return
        self.types_tainted = True
        self.tainted = True
        self.frozen = False
        for child in self.children:
            child.taint_children_type(force=force)
