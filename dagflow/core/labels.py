from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from ..tools.schema import LoadYaml

if TYPE_CHECKING:
    from collections.abc import Callable, Container, Sequence
    from typing import Any


def format_latex(k: str, s: str | Any, /, *args, protect_latex: bool = True, **kwargs) -> str:
    if not isinstance(s, str):
        return s

    if "{" not in s:
        return s

    should_protect_latex = protect_latex and k in {"latex", "axis", "xaxis", "rootaxis"}
    if should_protect_latex and "{" in s and "{{" not in s:  # }} }
        return s

    return s.format(*args, **kwargs)


def format_dict(
    dct: Mapping | str,
    /,
    *args,
    process_keys: tuple | list | set | None = None,
    protect_latex: bool = True,
    **kwargs,
) -> dict:
    if isinstance(dct, str):
        return {"text": format_latex("", dct, *args, **kwargs)}

    ret = {}
    for key, value in dct.items():
        match value:
            case dict():
                ret[key] = format_dict(
                    value, *args, process_keys=process_keys, protect_latex=protect_latex, **kwargs
                )
            case str():
                if process_keys and key not in process_keys:
                    continue
                ret[key] = format_latex(key, value, *args, protect_latex=protect_latex, **kwargs)
            case _:
                ret[key] = value

    return ret


def mapping_append_lists(dct: dict, key: str, lst: list):
    if not isinstance(dct, dict):
        return

    def patch(dct):
        oldlist = dct.get(key, [])
        newlist = list(lst)
        for name in oldlist:
            if name not in newlist:
                newlist.append(name)
        dct[key] = newlist

    has_subdicts = False
    for v in dct.values():
        if isinstance(v, dict):
            mapping_append_lists(v, key, lst)
            has_subdicts = True

    if not has_subdicts:
        patch(dct)


def repr_pretty(self, p, cycle):
    """Pretty repr for IPython.

    To be used as __repr__ method
    """
    p.text("..." if cycle else str(self))


def _make_formatter(fmt: str | Callable | dict | None) -> Callable:
    if isinstance(fmt, str):
        return fmt.format
    elif isinstance(fmt, dict):

        def formatter(s, **_):
            return fmt.get(s, s)

        return formatter
    elif fmt is None:

        def formatter(s, **_):
            return s

        return formatter

    return fmt


class Labels:
    __slots__ = (
        "_name",
        "_index_values",
        "_index_dict",
        "_text",       # for terminal
        "_graph",      # for the graph
        "_latex",      # for latex output, or to replace plottitle
        "_mark",       # for short mark on the graphiz graph
        "_xaxis",      # when object is used as X axis for other object
        "_axis",       # for the relevant axis, will be replaced with plottitle if not found
        "_plottitle",  # for plot title, will be replaced by latex if not found
        "_roottitle",  # for canvas title (root), will be replaced by plottitle with \→# substitution
        "_rootaxis",
        "_paths",
        "_plotmethod",
        "_node_hidden",
    )

    _name: str | None
    _index_values: list[str]
    _index_dict: dict[str, tuple[str, int]]
    _text: str | None
    _graph: str | None
    _latex: str | None
    _axis: str | None
    _xaxis: str | None
    _plottitle: str | None
    _roottitle: str | None
    _rootaxis: str | None
    _mark: str | None
    _paths: list[str]
    _plotmethod: str | None
    _node_hidden: bool | None

    def __init__(self, label: dict[str, str] | str | Path | None = None):
        for slot in self.__slots__:
            setattr(self, slot, None)
        self._paths = []
        self._index_values = []
        self._index_dict = {}
        self._node_hidden = False

        if isinstance(label, str):
            if label.endswith(".yaml"):
                self._update_from(label)
            else:
                self._text = label
        elif isinstance(label, Path):
            self._update_from(str(label))
        elif isinstance(label, dict):
            self.update(label)

    def __str__(self):
        return str(
            {
                slot.removeprefix("_"): v
                for slot in self.__slots__
                if (v := getattr(self, slot)) is not None
            }
        )

    _repr_pretty_ = repr_pretty

    def _update_from(self, path: str):
        d = LoadYaml(path)
        self.update(d)

    def update(self, d: dict[str, str | dict[str, str]]):
        match d:
            case {"group": {} as group, **rest} if not rest:
                d = group
                for k, v in d.items():
                    d[k] = v.format(space_key="", key_space="", key="", index=())

        for k, v in d.items():
            setattr(self, k, v)

    def format(self, *args, **kwargs):
        for name in (
            "text",
            "graph",
            "latex",
            "axis",
            "xaxis",
            "plottitle",
            "roottitle",
            "rootaxis",
        ):
            aname = f"_{name}"
            oldvalue = getattr(self, aname)
            newvalue = format_latex(name, oldvalue, *args, **kwargs)
            setattr(self, aname, newvalue)

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def index_dict(self) -> dict[str, tuple[str, int]]:
        return self._index_dict

    def index_in_mask(
        self, accepted_items: Mapping[str, str | int | Container[str | int]] | None
    ) -> bool:
        if accepted_items is None:
            return True

        index = self.index_dict
        for category, accepted_list in accepted_items.items():
            if (idxnum := index.get(category)) is None:
                continue

            if isinstance(accepted_list, str):
                if (
                    idxnum[0] != accepted_list  # key value
                    and idxnum[1] != accepted_list  # key index
                ):
                    return False
            elif (
                idxnum[0] not in accepted_list  # key value
                and idxnum[1] not in accepted_list  # key index
            ):
                return False

        return True

    @index_dict.setter
    def index_dict(self, index_dict: dict[str, tuple[str, int]]):
        self._index_dict = index_dict

    def build_index_dict(self, index: Mapping[str, Sequence[str]] = {}):
        if not index or self.index_dict:
            return

        if not self.index_values and self.paths:
            path = self.paths[0]
            self.index_values = path.split(".")

        to_remove = []
        index_values = self.index_values
        for value in index_values:
            for category, possible_values in index.items():
                try:
                    pos = possible_values.index(value)
                except ValueError:
                    continue
                else:
                    self.index_dict[category] = value, pos
                    break
            else:
                to_remove.append(value)
        for v in to_remove:
            index_values.remove(v)

    @property
    def index_values(self) -> list[str]:
        return self._index_values

    @index_values.setter
    def index_values(self, index_values: list[str]):
        self._index_values = list(index_values)

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def text(self) -> str | None:
        return self._text

    @text.setter
    def text(self, value: str):
        self._text = value

    @property
    def graph(self) -> str | None:
        return self._graph or self._text

    @graph.setter
    def graph(self, value: str | None):
        self._graph = value

    @property
    def latex(self) -> str | None:
        return self._latex or self._text

    @latex.setter
    def latex(self, value: str):
        self._latex = value

    @property
    def plottitle(self) -> str | None:
        return self._plottitle or self._latex or self._text

    @plottitle.setter
    def plottitle(self, value: str | None):
        self._plottitle = value

    @property
    def roottitle(self) -> str | None:
        if self._roottitle is not None:
            return self._roottitle
        return _latex_to_root(self.plottitle)

    @roottitle.setter
    def roottitle(self, value: str | None):
        self._roottitle = value

    @property
    def rootaxis(self) -> str | None:
        return _latex_to_root(self.axis) if self._rootaxis is None else self._rootaxis

    @rootaxis.setter
    def rootaxis(self, value: str | None):
        self._rootaxis = value

    @property
    def axis(self) -> str | None:
        return self._axis or self.plottitle

    @axis.setter
    def axis(self, value: str | None):
        self._axis = value

    @property
    def xaxis(self) -> str | None:
        return self._xaxis

    @xaxis.setter
    def xaxis(self, value: str | None):
        self._xaxis = value

    @property
    def mark(self) -> str | None:
        return self._mark

    @mark.setter
    def mark(self, value: str):
        self._mark = value

    @property
    def path(self) -> str | None:
        try:
            return self._paths[0]
        except IndexError:
            return None

    @property
    def paths(self) -> list[str]:
        return self._paths

    @paths.setter
    def paths(self, value: list[str]):
        self._paths = value

    @property
    def plottable(self) -> bool:
        return self._plotmethod != "none"

    @property
    def plotmethod(self) -> str | None:
        return self._plotmethod

    @plotmethod.setter
    def plotmethod(self, value: str):
        self._plotmethod = value.lower()

    @property
    def node_hidden(self) -> bool:
        return bool(self._node_hidden)

    @node_hidden.setter
    def node_hidden(self, value: bool):
        self._node_hidden = value

    def items(self):
        for k in self.__slots__:
            yield k, getattr(self, k)

    def __getitem__(self, k: str) -> str | None:
        return getattr(self, k)

    def __setitem__(self, k: str, v: str) -> str | None:
        return setattr(self, k, v)

    def get(self, k: str, default: str) -> str | None:
        return getattr(self, k, default)

    def setdefault(self, k: str, default: str) -> str | None:
        if (value := getattr(self, k, None)) is not None:
            return value

        setattr(self, k, default)
        return default

    def setdefaults(self, labels: dict):
        for k, v in labels.items():
            if getattr(self, k) is None:
                setattr(self, k, v)

    def copy(self) -> Labels:
        l = Labels()
        for slot in self.__slots__:
            setattr(l, slot, getattr(self, slot))
        return l

    def inherit(
        self,
        source: Labels | Mapping,
        fmtlong: str | Callable | None = None,
        fmtshort: str | Callable | None = None,
        fields: Sequence[str] = [],
        fields_exclude: Container[str] = [],
        fmtextra: Mapping[str, str] = {},
    ):
        fmtlong = _make_formatter(fmtlong)
        fmtshort = _make_formatter(fmtshort)

        if fields:
            inherit = tuple(f"_{s}" for s in fields)
        else:
            inherit = (
                "_text",
                "_graph",
                "_latex",
                "_mark",
                "_axis",
                "_plottitle",
                "_index_values",
                "_index_dict",
                "_paths",
                "_node_hidden",
            )
        kshort = {"_mark"}
        for _key in inherit:
            if _key[1:] in fields_exclude:
                continue
            if isinstance(source, Labels):
                label = getattr(source, _key, None)
            elif isinstance(source, Mapping):
                key = _key[1:]
                label = source.get(key, None)
            else:
                raise ValueError(source)
            if label is None:
                continue
            match label:
                case str():
                    formatter = fmtshort if _key in kshort else fmtlong
                    newv = formatter(label, source=source)
                    if newv is not None:
                        self[_key] = newv
                case tuple() | {} | []:
                    self[_key] = type(label)(label)
                case _:
                    self[_key] = label

        for key, fmt in fmtextra.items():
            _key = f"_{key}"
            if getattr(self, _key) is not None:
                continue

            formatter = _make_formatter(fmt)
            self[_key] = formatter(source=source)


def inherit_labels(
    source: dict,
    destination: dict | None = None,
    *,
    fmtlong: str | Callable,
    fmtshort: str | Callable,
    fields_exclude: Container[str] = [],
) -> dict:
    if destination is None:
        destination = {}

    fmtlong = _make_formatter(fmtlong)
    fmtshort = _make_formatter(fmtshort)

    kshort = {"mark"}
    kskip = {"key", "name"}
    for k, v in source.items():
        if k in fields_exclude:
            continue
        if k in kskip:
            continue
        if isinstance(v, str):
            newv = fmtshort(v) if k in kshort else fmtlong(v)
            if newv is not None:
                destination[k] = newv
        else:
            destination[k] = v

    return destination


def _latex_to_root(text: str | None) -> str | None:
    return text.replace(r"\rm ", "").replace("\\", "#").replace("$", "") if text else text
