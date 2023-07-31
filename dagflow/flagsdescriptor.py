class FlagsDescriptor:
    """
    The class stores the node flags
    """

    tainted: bool
    frozen: bool
    frozen_tainted: bool
    invalid: bool
    closed: bool
    allocated: bool
    being_evaluated: bool
    types_tainted: bool

    __slots__ = (
        "tainted",
        "frozen",
        "frozen_tainted",
        "invalid",
        "closed",
        "allocated",
        "being_evaluated",
        "types_tainted",
    )

    def __init__(
        self,
        *,
        tainted: bool = True,
        frozen: bool = False,
        frozen_tainted: bool = False,
        invalid: bool = False,
        closed: bool = False,
        allocated: bool = False,
        being_evaluated: bool = False,
        types_tainted: bool = True,
    ) -> None:
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
