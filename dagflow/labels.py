from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path

from .tools.schema import LoadYaml


def format_latex(k, s, /, *args, **kwargs) -> str:
    if not isinstance(s, str):
        return s
    if (k=='latex' and '$' in s) or '{' not in s:
        return s

    return s.format(*args, **kwargs)

def repr_pretty(self, p, cycle):
    """Pretty repr for IPython. To be used as __repr__ method"""
    p.text(str(self) if not cycle else "...")

def _make_formatter(fmt: str | Callable | dict | None) -> Callable:
    if isinstance(fmt, str):
        return fmt.format
    elif isinstance(fmt, dict):
        return lambda s: fmt.get(s, s)
    elif fmt is None:
        return lambda s: s

    return fmt

class Labels:
    __slots__ = (
        "_name",
        "_index_values",
        "_index_dict",
        "_text",
        "_graph",
        "_latex",
        "_mark",
        "_xaxis",
        "_axis",
        "_plottitle",
        "_roottitle",
        "_rootaxis",
        "_paths",
        "_plotmethod"
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

    def __init__(self, label: dict[str, str] | str | Path | None=None):
        for slot in self.__slots__:
            setattr(self, slot, None)
        self._paths = []
        self._index_values = []
        self._index_dict = {}

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
        return str({
            slot.removeprefix("_"): v
            for slot in self.__slots__
            if (v:=getattr(self, slot)) is not None
        })

    _repr_pretty_ = repr_pretty

    def _update_from(self, path: str):
        d = LoadYaml(path)
        self.update(d)

    def update(self, d: dict[str,str]):
        for k, v in d.items():
            setattr(self, k, v)

    def format(self, *args, **kwargs):
        for name in (
                "text", "graph", "latex",
                "xaxis", "plottitle", "roottitle",
                "rootaxis"
                ):
            aname = f"_{name}"
            oldvalue = getattr(self, aname)
            newvalue = format_latex(name, oldvalue, *args, **kwargs)
            setattr(self, aname, newvalue)

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def index_dict(self) -> dict[str,tuple[str,int]]:
        return self._index_dict

    @index_dict.setter
    def index_dict(self, index_dict: dict[str,tuple[str,int]]):
        self._index_dict = index_dict

    def build_index_dict(self, index: Mapping[str, Sequence[str]]={}):
        if not index or self.index_dict:
            return

        if not self.index_values and self.paths:
            path = self.paths[0]
            self.index_values = path.split('.')

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
        return self._latex

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
        if self._rootaxis is not None:
            return self._rootaxis
        return _latex_to_root(self.axis)

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
        return self._plotmethod!="none"

    @property
    def plotmethod(self) -> str | None:
        return self._plotmethod

    @plotmethod.setter
    def plotmethod(self, value: str):
        self._plotmethod = value.lower()

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
        if (value:=getattr(self, k, None)) is not None:
            return value

        setattr(self, k, default)
        return default

    def setdefaults(self, labels: dict):
        for k, v in labels.items():
            if getattr(self, k) is None:
                setattr(self, k, v)

    def copy(self) -> "Labels":
        l = Labels()
        for slot in self.__slots__:
            setattr(l, slot, getattr(self, slot))
        return l

    def inherit(
        self,
        source: "Labels",
        fmtlong: str | Callable | None=None,
        fmtshort: str | Callable | None=None,
        fields: Sequence[str] = []
    ):
        fmtlong = _make_formatter(fmtlong)
        fmtshort = _make_formatter(fmtshort)

        if fields:
            inherit = tuple(f'_{s}' for s in fields)
        else:
            inherit = (
                "_text",
                "_graph",
                "_latex",
                "_mark",
                "_axis",
                "_plottitle",
                '_index_values',
                '_index_dict'
            )
        kshort = {"_mark"}
        for _key in inherit:
            label = getattr(source, _key, None)
            if label is None: continue
            match label:
                case str():
                    newv = fmtshort(label) if _key in kshort else fmtlong(label)
                    if newv is not None:
                        self[_key] = newv
                case tuple() | dict() | list():
                    self[_key] = type(label)(label)
                case _:
                    self[_key] = label

def inherit_labels(
        source: dict,
        destination: dict | None=None,
        *,
        fmtlong: str | Callable,
        fmtshort: str | Callable
) -> dict:
    if destination is None:
        destination = {}

    fmtlong = _make_formatter(fmtlong)
    fmtshort = _make_formatter(fmtshort)

    kshort = {"mark"}
    kskip = {"key", "name"}
    for k, v in source.items():
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
    if not text:
        return text
    return text.replace(r"\rm ", "").replace("\\", "#").replace("$","")
