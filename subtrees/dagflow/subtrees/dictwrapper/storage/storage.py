from __future__ import annotations

from collections import UserDict
from collections.abc import Sequence
from typing import Any, Callable, Generator, Optional

class Storage(UserDict):
    _protect: bool = False

    def __init__(*args, protect: bool = False, **kwargs) -> None:
        self = args[0]
        self._protect = protect
        UserDict.__init__(*args, **kwargs)

    def _process_key(self, key: Any) -> tuple:
        if isinstance(key, Sequence):
            return tuple(sorted(key))
        else:
            return frozenset((key,))

    def __getitem__(self, key: Any) -> Any:
        key = self._process_key(key)
        return super().__getitem__(key)

    def __setitem__(self, key: Any, val: Any) -> None:
        key = self._process_key(key)
        if self._protect and key in self:
            raise AttributeError(
                f"Reassigning of the existed key '{key}' is restricted, "
                "due to the protection!"
            )
        super().__setitem__(key, val)

    def __contains__(self, key: Any) -> bool:
        key = self._process_key(key)
        return super().__contains__(key)

    def values(self, *, keys: tuple = (), **kwargs) -> Generator:
        for _, val in self.items(*keys, **kwargs):
            yield val

    def keys(self, *args, **kwargs) -> Generator:
        for key, _ in self.items(*args, **kwargs):
            yield key

    def items(
        self,
        *args,
        filterkey: Optional[Callable[[Any], bool]] = None,
        filterkeyelem: Optional[Callable[[Any], bool]] = None,
    ) -> tuple:
        """
        Returns items from the slice by `args`.
        If `args` are empty returns all items.
        """
        res = super().items()
        if args:
            args = set(args)
            res = (elem for elem in res if args.issubset(elem[0]))
        if filterkey:
            res = (elem for elem in res if filterkey(elem[0]))
        if filterkeyelem:
            res = (
                elem
                for elem in res
                if all(filterkeyelem(key) for key in elem[0])
            )

        yield from res

    def slice(self, *args, **kwargs) -> Storage:
        """
        Returns new `Storage` with keys containing `args`.
        It is possible to filter elements by `filterkey` and `filterkeyelem`.
        """
        return Storage(
            self.items(
                *args,
                filterkey=kwargs.pop("filterkey", None),
                filterkeyelem=kwargs.pop("filterkeyelem", None),
            ),  # type: ignore
            **kwargs,
        )
