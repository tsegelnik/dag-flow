from __future__ import annotations

from collections import UserDict
from itertools import product
from typing import Any, Iterator, Literal, Optional, Sequence, Tuple, Union

from attr import Factory, define, field
from attr.validators import instance_of

from .gindex import GIndex, GIndexInstance, GIndexName, namemode_validator


class GIndexNameDict(UserDict):
    """
    The same as usual dict, but keys are `GIndexName` objects.
    It is possible to use a `GIndexName.short` or `GIndexName.full`,
    instead of the `GIndexName` object itself.
    In the `set` method new `GIndexName(key, key)` will be created
    at usage of `key` (`type=str`) if there is no existing `GIndexName`
    object:
    `obj["det"] = val` the same as `obj[GIndexName("det", "det")] = val`
    """

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            for elem in self:
                if key in elem:
                    return self.data[elem]
        return super().__getitem__(key)

    def __setitem__(self, key: Any, val: Any) -> None:
        if isinstance(key, str):
            for elem in self:
                if key in elem:
                    self.data[elem] = val
                    return
            self[GIndexName(key, key)] = val
        else:
            super().__setitem__(key, val)

    def __delitem__(self, key: Any) -> None:
        if isinstance(key, str):
            for elem in self:
                if key in elem:
                    del self.data[elem]
                    return
            raise KeyError()
        super().__delitem__(key)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str):
            return key in (name for names in self for name in names)
        return super().__contains__(key)


@define(hash=True, slots=True)
class GNIndexInstance:
    """
    The n-dimensional index instance class, storing `indices`
    (`type=list[GIndexInstance]`) and `names` (`type=dict[GIndexName, ]`).
    Contains `format` method, which substitutes `value` instead of `name.short`
    and `name.full`.

    :param order: The tuple of the `GIndexInstance`s names
        (`type=str`, use `name.short` or `name.full`) and any `int`,
        that is the `string` for formatting
    """

    _indices: Tuple[GIndexInstance, ...] = field(default=tuple(), alias='indices')
    order: tuple = field(default=tuple())
    sep: str = field(validator=instance_of(str), default="_")
    withname: bool = field(validator=instance_of(bool), default=False)
    namemode: Literal["s", "f", "short", "full"] = field(
        validator=namemode_validator, default="s"
    )
    namesep: str = field(validator=instance_of(str), default="")
    dict: GIndexNameDict = field(
        default=Factory(lambda self: self._create_dict(), takes_self=True),
        repr=False,
    )

    @property
    def values(self) -> Tuple[str]:
        return tuple(instance.value for instance in self._indices)

    def __attrs_post_init__(self) -> None:
        self._check_indices()
        if not self.order:
            self.order = self._auto_order()
        else:
            self._check_order(self.order)
        self.sort()

    def sort(
        self, order: Optional[tuple] = None, rest2end: bool = True
    ) -> None:
        if not order:
            order = self.order
        indices = [self.dict[name] for name in order if name in self.dict]
        if len(self._indices) != len(indices):
            names = set(self.dict.keys()) - set(order)
            for name in names:
                if rest2end:
                    indices.append(self.dict[name])
                else:
                    indices.insert(0, self.dict[name])
        self._indices = tuple(indices)

    def _create_dict(self) -> GIndexNameDict:
        return GIndexNameDict({val.name: val for val in self._indices})

    def _auto_order(self) -> tuple:
        return (True,) + tuple(val.name.s for val in self._indices)

    def _check_order(self, order: Sequence) -> None:
        if not isinstance(order, Sequence):
            raise TypeError(
                f"'order' must be `Sequence`, but given '{type(order)}'!"
            )
        elif not isinstance(order, tuple):
            order = tuple(order)

    def _check_indices(self) -> None:
        if not isinstance(self._indices, (Sequence, set)):
            raise TypeError(
                f"'indices' must be `Sequence`, but given '{type(self._indices)}'!"
            )
        elif not all(isinstance(x, GIndexInstance) for x in self._indices):
            raise ValueError(
                "'indices' must be `Sequence[GIndexInstance]`, "
                f"but given '{self._indices}'!"
            )
        elif not isinstance(self._indices, tuple):
            self._indices = tuple(self._indices)

    def formatwith(
        self,
        string: str,
        withname: bool = False,
        namemode: Literal["s", "f", "short", "full"] = "s",
        sep: Optional[str] = None,
        namesep: Optional[str] = None,
        order: Optional[tuple] = None,
    ) -> str:
        if not isinstance(string, str):
            raise TypeError(
                f"'string' must be 'str', but given {type(string)}!"
            )
        if not order:
            order = self.order
        else:
            self._check_order(order)
        if not sep:
            sep = self.sep
        if not namesep:
            namesep = self.namesep
        ordered = []
        for name in order:
            if name in self.dict:
                ordered.append(
                    self.dict[name].formattedwith(
                        sep=sep,
                        withname=withname,
                        namemode=namemode,
                        namesep=namesep,
                    )
                )
            elif isinstance(name, int):
                ordered.append(string)
        return "".join(ordered)

    def formattedwith(
        self,
        withname: bool = False,
        namemode: Literal["s", "f", "short", "full"] = "s",
        sep: Optional[str] = None,
        namesep: Optional[str] = None,
        order: Optional[tuple] = None,
    ) -> str:
        if not order:
            order = self.order
        else:
            self._check_order(order)
        if not sep:
            sep = self.sep
        if not namesep:
            namesep = self.namesep
        return "".join(
            self.dict[name].formattedwith(
                withname=withname, namemode=namemode, sep=sep, namesep=namesep
            )
            for name in order
            if name in self.dict
        )

    def format(self, string: str) -> str:
        if not isinstance(string, str):
            raise TypeError(
                f"'string' must be 'str', but given {type(string)}!"
            )
        ordered = []
        for name in self.order:
            if name in self.dict:
                ordered.append(self.dict[name].formatted())
            elif name == True:
                ordered.append(string)
        return "".join(ordered)

    def formatted(self) -> str:
        return "".join(
            self.dict[name].formatted()
            for name in self.order
            if name in self.dict
        )

    def size(self) -> int:
        """Returns the size of the list with indices (number of variants)"""
        return len(self._indices)

    def copy(self, deep: bool = False) -> GNIndexInstance:
        """Returns a copy of the object"""

        if deep:
            ret = GNIndexInstance(
                indices=tuple(self._indices),
                order=tuple(self.order),
                sep=str(self.sep),
                withname=bool(self.withname),
                namemode=str(self.namemode),  # type: ignore
                namesep=str(self.namesep),
            )
        else:
            ret = GNIndexInstance(
                indices=self._indices,
                order=self.order,
                sep=self.sep,
                withname=self.withname,
                namemode=self.namemode,
                namesep=self.namesep,
            )

        if kwargs:
            raise RuntimeError(f"GNIndexInstance.copy() unparsed arguments: {kwargs}")

        return ret

    def copywith(self, **kwargs) -> GNIndexInstance:
        """Returns a copy of the object with updated fields from `kwargs`"""
        if kwargs.pop("deep", True):
            ret = GNIndexInstance(
                indices=kwargs.pop("indices", tuple(self._indices)),
                order=kwargs.pop("order", tuple(self.order)),
                sep=kwargs.pop("sep", str(self.sep)),
                withname=kwargs.pop("withname", bool(self.withname)),
                namemode=kwargs.pop("namemode", str(self.namemode)),
                namesep=kwargs.pop("namesep", str(self.namesep)),
            )
        else:
            ret = GNIndexInstance(
                indices=kwargs.pop("indices", self._indices),
                order=kwargs.pop("order", self.order),
                sep=kwargs.pop("sep", self.sep),
                withname=kwargs.pop("withname", self.withname),
                namemode=kwargs.pop("namemode", self.namemode),
                namesep=kwargs.pop("namesep", self.namesep),
            )

        if kwargs:
            raise RuntimeError(f"GNIndexInstance.copywith() unparsed arguments: {kwargs}")

        return ret

    def __iter__(self) -> Iterator[GIndexInstance]:
        yield from self._indices

    def __getitem__(self, key: int) -> GIndexInstance:
        if not isinstance(key, int):
            raise TypeError(f"'key' must be 'int', but given '{type(key)}'!")
        return self._indices[key]


@define(hash=True, slots=True)
class GNIndex:
    """
    The n-dimensional index class, storing the `indices`
    (set of the 1-dim indices), the indices `order` and usefull methods
    """

    _indices: Tuple[GIndex] = field(default=tuple(), alias='indices')
    order: tuple = field(default=tuple())
    sep: str = field(validator=instance_of(str), default="_")
    withname: bool = field(validator=instance_of(bool), default=False)
    namemode: Literal["s", "f", "short", "full"] = field(
        validator=namemode_validator, default="s"
    )
    namesep: str = field(validator=instance_of(str), default="")
    dict: GIndexNameDict = field(
        default=Factory(lambda self: self._create_dict(), takes_self=True),
        repr=False,
    )

    @staticmethod
    def from_dict(data: dict) -> "GNIndex":
        return GNIndex(
                tuple(
                    GIndex(GIndexName(name, name), values) if isinstance(name, str) \
                    else GIndex(GIndexName(*name), values) \
                    for name, values in data.items()
                    )
                )

    def __attrs_post_init__(self) -> None:
        self._check_indices()
        if not self.order:
            self.order = self._auto_order()
        else:
            self._check_order(self.order)
        self.sort()

    def _auto_order(self) -> tuple:
        return (True,) + tuple(val.name.s for val in self._indices)

    def _check_order(self, order: Sequence) -> None:
        if not isinstance(order, Sequence):
            raise TypeError(
                f"'order' must be `Sequence`, but given '{type(order)}'!"
            )
        elif not isinstance(order, tuple):
            order = tuple(order)

    def _check_indices(self) -> None:
        if not isinstance(self._indices, (Sequence, set)):
            raise TypeError(
                f"'indices' must be `Sequence`, but given '{type(self._indices)}'!"
            )
        elif not all(isinstance(x, GIndex) for x in self._indices):
            raise ValueError(
                "'indices' must be `Sequence[GIndex]`, "
                f"but given '{self._indices}'!"
            )
        elif not isinstance(self._indices, tuple):
            self._indices = tuple(self._indices)

    def _create_dict(self) -> GIndexNameDict:
        return GIndexNameDict({val.name: val for val in self._indices})

    def rest(
        self,
        names: Union[
            str,
            GIndexName,
            Sequence[Union[str, GIndexName, Sequence[Union[str, GIndexName]]]],
        ],
    ) -> Optional[GNIndex]:
        """
        Returns rest indices from `names`.

        param names: A `str`, or `Sequence[str]`,
        or `Sequence[Sequence[str]]` of indices names,
        which are will be used to find the rest indices.
        It is possible to use `GIndexName`, instead of `str`

        return: A `GNIndex` with tuple of the rest indices
        """
        if len(self._indices) == 0:
            return None
        if isinstance(names, (list, tuple, set)):
            return (
                self.copywith(indices=indices)
                if len(names) != 0
                and (
                    indices := tuple(
                        self._rest(self.dict.copy(), names).values()
                    )
                )
                else None
            )
        elif isinstance(names, (str, GIndexName)):
            tmpdict = self.dict.copy()
            tmpdict.pop(names)
            return self.copywith(indices=tuple(tmpdict.values()))
        raise TypeError(
            f"'names' must be `Sequence[str]`, but given '{type(names)}'!"
        )

    def _rest(
        self,
        tmpdict: GIndexNameDict,
        names: Sequence[
            Union[str, GIndexName, Sequence[Union[str, GIndexName]]]
        ],
    ) -> GIndexNameDict:
        for name in names:
            if isinstance(name, (str, GIndexName)):
                tmpdict.pop(name)
            elif isinstance(name, (list, tuple, set)):
                self._rest(tmpdict, name)
        return tmpdict

    def split(
        self,
        names: Sequence[
            Union[str, GIndexName, Sequence[Union[str, GIndexName]]]
        ],
        rest: bool = True,
    ) -> tuple:
        if not isinstance(names, (list, tuple, set)):
            raise TypeError(
                "'names' must be Sequence[str, GIndexName], "
                f"but given {type(names)}!"
            )
        return (
            self._split(names),
            (self.rest(names) if rest else None),
        )

    # TODO: are we need empty GNIndex or None?
    #       if empty, it is no error while iter
    def _split(
        self,
        names: Sequence[
            Union[Sequence[Union[str, GIndexName]], str, GIndexName]
        ],
    ) -> GNIndex:
        res = []
        if isinstance(names, (list, tuple, set)):
            res.extend(self.__split(name) for name in names)
        else:
            res.append(self.__split(names))
        return self.copywith(indices=tuple(res))

    def __split(self, name: Any) -> GIndex:
        if not isinstance(name, (str, GIndexName)):
            raise TypeError(
                "It is possible split only by 2D Sequence[str, GIndexName]!"
            )
        if elem := self.dict.get(name, False):
            return elem
        raise ValueError(f"There is no index with name '{name}'!")

    def sub(self, names: tuple) -> GNIndex:
        return self.copywith(
            indices=tuple(
                val
                for val in self._indices
                if (val.name.s in names or val.name.f in names)
            )
        )

    subindex = sub

    def union(self, *args, **kwargs) -> GNIndex:
        indices = [*self._indices]
        for arg in args:
            if not isinstance(arg, GNIndex):
                raise TypeError(
                    f"'args' must be `GNIndex`, but given '{type(arg)}'"
                )

            indices.extend(index for index in arg._indices if index not in indices)
        return self.copywith(indices=indices, **kwargs)

    def __add__(self, right: GNIndex) -> GNIndex:
        if not isinstance(right, GNIndex):
            raise TypeError(
                f"'right' must be `GNIndex`, but given '{type(right)}'"
            )
        elif self.order != right.order:
            raise AttributeError(
                "'right' must have the same `order` as the left,"
                f"but given '{self.order=}', '{right.order=}'"
            )
        return self.copywith(indices=set(self._indices + right._indices))

    def __or__(self, right: GNIndex) -> GNIndex:
        return self.__add__(right)

    def __sub__(self, right: GNIndex) -> GNIndex:
        if not isinstance(right, GNIndex):
            raise TypeError(
                f"'right' must be `GNIndex`, but given '{type(right)}'"
            )
        elif self.order != right.order:
            raise AttributeError(
                "'right' must have the same `order` as the left,"
                f"but given '{self.order=}', '{right.order=}'"
            )
        return self.copywith(indices=set(self._indices) - set(right._indices))

    def __xor__(self, right: GNIndex) -> GNIndex:
        return self.__sub__(right)

    def sort(self, order: Optional[tuple] = None) -> None:
        if not order:
            order = self.order
        tmpdict = self.dict.copy()
        indices = [tmpdict.pop(name) for name in order if name in tmpdict]
        if idxs := tmpdict.values():
            indices.extend(idxs)
        self._indices = tuple(indices)

    def dim(self) -> int:
        """Returns the dimension of the index (size of the indices list)"""
        return len(self._indices)

    @property
    def values(self) -> Tuple[Tuple[str]]:
        return tuple(n.values for n in self)

    def instances(self) -> Tuple[Tuple[GNIndexInstance, ...], ...]:
        """Returns a tuple of the indices instances tuples (2D version)"""
        return tuple(ind.instances() for ind in self._indices)

    def instances1d(self) -> Tuple[GNIndexInstance, ...]:
        """Returns a tuple of the indices instances (1D version)"""
        return tuple(inst for ind in self._indices for inst in ind.instances())

    def names(self) -> tuple:
        return tuple(val.name for val in self._indices)

    def names1d(
        self, namemode: Literal["s", "f", "short", "full"] = "s"
    ) -> tuple:
        return tuple(val.name[namemode] for val in self._indices)

    def __iter__(self) -> Iterator[GNIndexInstance]:
        for val in product(*self.instances()):
            yield GNIndexInstance(
                indices=val,  # type:ignore
                order=self.order,
                sep=self.sep,
                withname=self.withname,
                namemode=self.namemode,
                namesep=self.namesep,
            )

    def copy(self, deep: bool = False) -> GNIndex:
        """Returns a copy of the object"""
        if deep:
            ret = GNIndex(
                indices=tuple(self._indices),
                order=tuple(self.order),
                sep=str(self.sep),
                withname=bool(self.withname),
                namemode=str(self.namemode),  # type:ignore
                namesep=str(self.namesep),
            )
        else:
            ret = GNIndex(
                indices=self._indices,
                order=self.order,
                sep=self.sep,
                withname=self.withname,
                namemode=self.namemode,
                namesep=self.namesep,
            )

        if kwargs:
            raise RuntimeError(f"GNIndex.copy() unparsed arguments: {kwargs}")

        return ret

    def copywith(self, **kwargs) -> GNIndex:
        """Returns a copy of the object with updated fields from `kwargs`"""

        if kwargs.pop("deep", True):
            ret = GNIndex(
                indices=kwargs.pop("indices", tuple(self._indices)),
                order=kwargs.pop("order", tuple(self.order)),
                sep=kwargs.pop("sep", str(self.sep)),
                withname=kwargs.pop("withname", bool(self.withname)),
                namemode=kwargs.pop("namemode", str(self.namemode)),
                namesep=kwargs.pop("namesep", str(self.namesep)),
            )
        else:
            ret = GNIndex(
                indices=kwargs.pop("indices", self._indices),
                order=kwargs.pop("order", self.order),
                sep=kwargs.pop("sep", self.sep),
                withname=kwargs.pop("withname", self.withname),
                namemode=kwargs.pop("namemode", self.namemode),
                namesep=kwargs.pop("namesep", self.namesep),
            )

        if kwargs:
            raise RuntimeError(f"GNIndex.copywith() unparsed arguments: {kwargs}")

        return ret
