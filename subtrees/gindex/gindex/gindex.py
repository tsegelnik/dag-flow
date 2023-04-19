from __future__ import annotations

from typing import Iterator, Literal, Optional, Sequence, Tuple
from warnings import warn

from attr import define, field
from attr.validators import instance_of


@define(hash=True, slots=True)
class GIndexName:
    """
    The index name class.
    Contains two fields: `s` ( or `short`) and `f` (or `full`).
    If `full` is not given, sets `full=short`.
    The fields are validated by `attr.validators` on the `str` type.
    """

    short: str = field(validator=instance_of(str))
    full: str = field(validator=instance_of(str), default="")

    @property
    def s(self):
        return self.short

    @s.setter
    def s(self, val):
        self.short = val

    @s.deleter
    def s(self):
        del self.short

    @property
    def f(self):
        return self.full

    @f.setter
    def f(self, val):
        self.full = val

    @f.deleter
    def f(self):
        del self.full

    def tuple(self) -> Tuple[str, str]:
        return (self.short, self.full)

    def __attrs_post_init__(self) -> None:
        if not self.full:
            self.full = self.short

    def copy(self, deep: bool = False) -> GIndexName:
        return (
            GIndexName(
                str(self.short),
                str(self.full),
            )
            if deep
            else GIndexName(
                self.short,
                self.full,
            )
        )

    def copywith(self, **kwargs) -> GIndexName:
        """Returns a copy of the object with updated fields from `kwargs`"""
        return (
            GIndexName(
                short=kwargs.pop("short", self.short),
                full=kwargs.pop("full", self.full),
            )
            if kwargs.pop("deep", False)
            else GIndexName(
                short=kwargs.pop("short", str(self.short)),
                full=kwargs.pop("full", str(self.full)),
            )
        )

    def __iter__(self) -> Iterator[str]:
        yield from self.tuple()

    def __getitem__(self, key: str) -> str:
        if key in {"s", "short"}:
            return self.short
        elif key in {"f", "full"}:
            return self.full
        else:
            raise ValueError(
                "'key' must be in ('s', 'f', 'short', 'full'), "
                f"but given {key}!"
            )

    def dict(self) -> dict:
        return {"short": self.short, "full": self.full}

    def __str__(self) -> str:
        return "{" + f"short: {self.short}, full: {self.full}" + "}"

    def __repr__(self) -> str:
        return self.__str__()


def namemode_validator(instance, attribute, value):
    if value not in {"s", "f", "short", "full"}:
        raise ValueError(
            "'namemode' must be in ('s', 'f', 'short', 'full'), "
            f"but given {value}!"
        )


@define(hash=True, slots=True)
class GIndexInstance:
    """
    The index instance class, storing a single `value` (`type=str`)
    and `name` (`type=GIndexName`).
    Contains `format` method, which substitutes `value` instead of `name.short`
    and `name.full`.
    """

    name: GIndexName = field(validator=instance_of(GIndexName))
    value: str = field(validator=instance_of(str))
    sep: str = field(validator=instance_of(str), default="_")
    withname: bool = field(validator=instance_of(bool), default=False)
    namemode: Literal["s", "f", "short", "full"] = field(
        validator=namemode_validator, default="s"
    )
    namesep: str = field(validator=instance_of(str), default="")
    _fmtstr: str = field(init=False, default="{sep}{indexname}{namesep}")

    def __attrs_post_init__(self) -> None:
        self._fmtstr += self.value
        if not self.withname and self.namesep:
            warn(
                "'namesep' is not used without 'withname=True'",
                RuntimeWarning,
            )

    def format(
        self,
        string: str,
        place: Optional[str] = None,
    ) -> str:
        """
        Formatting `string` with index `value`,
        using `name.short` and `name.full`
        """
        if not isinstance(string, str):
            raise TypeError(
                f"'string' must be 'str', but given {type(string)}!"
            )
        elif not string:
            return string
        return self._format(string=string, place=place)

    def _format(
        self,
        string: str,
        place: Optional[str],
    ) -> str:
        formatted = self.formatted()
        fmtdict = (
            {place: formatted}
            if place
            else {self.name.s: formatted, self.name.f: formatted}
        )
        return string.format(**fmtdict)

    def formatwith(
        self,
        string: str,
        withname: bool = False,
        namemode: Literal["s", "f", "short", "full"] = "s",
        sep: Optional[str] = None,
        namesep: Optional[str] = None,
        place: Optional[str] = None,
    ) -> str:
        """
        Formatting `string` with index `value`,
        using `name.short` and `name.full`
        """
        if not isinstance(string, str):
            raise TypeError(
                f"'string' must be 'str', but given {type(string)}!"
            )
        elif not string:
            return string
        return self._formatwith(
            string=string,
            withname=withname,
            namemode=namemode,
            sep=sep,
            namesep=namesep,
            place=place,
        )

    def _formatwith(
        self,
        string: str,
        withname: bool,
        namemode: Literal["s", "f", "short", "full"],
        sep: Optional[str],
        namesep: Optional[str],
        place: Optional[str],
    ) -> str:
        formatted = self.formattedwith(
            withname=withname, namemode=namemode, sep=sep, namesep=namesep
        )
        fmtdict = (
            {place: formatted}
            if place
            else {self.name.s: formatted, self.name.f: formatted}
        )
        return string.format(**fmtdict)

    def formatted(self) -> str:
        """
        Formatted index with default options
        """
        if self.withname:
            indexname = (
                self.name.s if self.namemode in ("s", "short") else self.name.f
            )
            namesep = self.namesep
        else:
            indexname = ""
            namesep = ""
        return self._fmtstr.format(
            sep=self.sep, indexname=indexname, namesep=namesep
        )

    def formattedwith(
        self,
        withname: bool = False,
        namemode: Literal["s", "f", "short", "full"] = "s",
        sep: Optional[str] = None,
        namesep: Optional[str] = None,
    ) -> str:
        """
        Formatted index with custom options
        """
        if sep is None:
            sep = self.sep
        elif not isinstance(sep, str):
            raise TypeError(f"'sep' must be 'str', but given {type(sep)}!")
        if withname:
            if namemode in {"s", "short"}:
                indexname = self.name.s
            elif namemode in {"f", "full"}:
                indexname = self.name.f
            else:
                raise ValueError(
                    "'namemode' must be in ('s', 'f', 'short', 'full'), "
                    f"but given '{namemode}'!"
                )
            if namesep is None:
                namesep = sep
            elif not isinstance(namesep, str):
                raise TypeError(
                    f"'namesep' must be 'str', but given {type(sep)}!"
                )
        else:
            indexname = ""
            if namesep:
                warn(
                    "'namesep' is not used without 'withname=True'",
                    RuntimeWarning,
                )
            namesep = ""
        return self._fmtstr.format(
            sep=sep, indexname=indexname, namesep=namesep
        )

    def copy(self, deep: bool = False) -> GIndexInstance:
        """Returns a copy of the object"""
        return (
            GIndexInstance(
                name=self.name.copy(),
                value=str(self.value),
                sep=str(self.sep),
                withname=bool(self.withname),
                namemode=str(self.namemode),  # type:ignore
                namesep=str(self.namesep),
            )
            if deep
            else GIndexInstance(
                name=self.name,
                value=self.value,
                sep=self.sep,
                withname=self.withname,
                namemode=self.namemode,
                namesep=self.namesep,
            )
        )

    def copywith(self, **kwargs) -> GIndexInstance:
        """Returns a copy of the object with updated fields from `kwargs`"""
        return (
            GIndexInstance(
                name=kwargs.pop("name", self.name.copy()),
                value=kwargs.pop("value", str(self.value)),
                sep=kwargs.pop("sep", str(self.sep)),
                withname=kwargs.pop("withname", bool(self.withname)),
                namemode=kwargs.pop("namemode", str(self.namemode)),
                namesep=kwargs.pop("namesep", str(self.namesep)),
            )
            if kwargs.pop("deep", True)
            else GIndexInstance(
                name=kwargs.pop("name", self.name),
                value=kwargs.pop("value", self.value),
                sep=kwargs.pop("sep", self.sep),
                withname=kwargs.pop("withname", self.withname),
                namemode=kwargs.pop("namemode", self.namemode),
                namesep=kwargs.pop("namesep", self.namesep),
            )
        )


@define(hash=True, slots=True)
class GIndex:
    """
    The index class, storing the `values`, `name` and usefull methods
    """

    name: GIndexName = field(validator=instance_of(GIndexName))
    values: tuple = field(default=tuple())
    sep: str = field(validator=instance_of(str), default="_")
    withname: bool = field(validator=instance_of(bool), default=False)
    namemode: Literal["s", "f", "short", "full"] = field(
        validator=namemode_validator, default="s"
    )
    namesep: str = field(validator=instance_of(str), default="")

    def __attrs_post_init__(self) -> None:
        if isinstance(self.values, tuple) and self.is_unique_values():
            return
        if isinstance(self.values, set):
            self.values = tuple(self.values)
        elif not self.is_unique_values():
            raise ValueError(f"'values' contains duplicates: {self.values}!")
        elif isinstance(self.values, list):
            self.values = tuple(self.values)
        else:
            raise TypeError(
                f"'values' must be `list`, `tuple` or `set` (got {self.values}"
                f"that is a {type(self.values)}!"
            )

    def __iter__(self) -> Iterator[GIndexInstance]:
        for val in self.values:
            yield GIndexInstance(
                name=self.name,
                value=val,
                sep=self.sep,
                withname=self.withname,
                namemode=self.namemode,
                namesep=self.namesep,
            )

    def instances(self) -> Sequence[GIndexInstance]:
        return tuple(self.__iter__())

    def __getitem__(self, key: int) -> GIndexInstance:
        if not isinstance(key, int):
            raise TypeError(f"'key' must be 'int', but given '{type(key)}'!")
        return GIndexInstance(
            name=self.name,
            value=self.values[key],
            sep=self.sep,
            withname=self.withname,
            namemode=self.namemode,
            namesep=self.namesep,
        )

    def size(self) -> int:
        """Returns the size of the list with values (number of variants)"""
        return len(self.values)

    def is_unique_values(self) -> bool:
        """Checks if the `values` contain only unique elements"""
        return len(self.values) == len(set(self.values))

    def copy(self, deep: bool = False) -> GIndex:
        """Returns a copy of the object"""
        return (
            GIndex(
                name=self.name.copy(),
                values=tuple(self.values),
                sep=str(self.sep),
                withname=bool(self.withname),
                namemode=str(self.namemode),  # type: ignore
                namesep=str(self.namesep),
            )
            if deep
            else GIndex(
                name=self.name,
                values=self.values,
                sep=self.sep,
                withname=self.withname,
                namemode=self.namemode,
                namesep=self.namesep,
            )
        )

    def copywith(self, **kwargs) -> GIndex:
        """Returns a copy of the object with updated fields from `kwargs`"""
        return (
            GIndex(
                name=kwargs.pop("name", self.name.copy()),
                values=kwargs.pop("values", tuple(self.values)),
                sep=kwargs.pop("sep", str(self.sep)),
                withname=kwargs.pop("withname", bool(self.withname)),
                namemode=kwargs.pop("namemode", str(self.namemode)),
                namesep=kwargs.pop("namesep", str(self.namesep)),
            )
            if kwargs.pop("deep", False)
            else GIndex(
                name=kwargs.pop("name", self.name),
                values=kwargs.pop("values", self.values),
                sep=kwargs.pop("sep", self.sep),
                withname=kwargs.pop("withname", self.withname),
                namemode=kwargs.pop("namemode", self.namemode),
                namesep=kwargs.pop("namesep", self.namesep),
            )
        )
