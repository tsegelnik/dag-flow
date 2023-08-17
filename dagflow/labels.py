from typing import Optional, Union, Callable, Dict, List, Tuple
from pathlib import Path
from .tools.schema import LoadYaml

def repr_pretty(self, p, cycle):
    """Pretty repr for IPython. To be used as __repr__ method"""
    p.text(str(self) if not cycle else "...")

def _make_formatter(fmt: Union[str, Callable, dict]) -> Callable:
    if isinstance(fmt, str):
        return fmt.format
    elif isinstance(fmt, dict):
        return lambda s: fmt.get(s, s)

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

    _name: Optional[str]
    _index_values: List[str]
    _index_dict: Dict[str, Tuple[str, int]]
    _text: Optional[str]
    _graph: Optional[str]
    _latex: Optional[str]
    _axis: Optional[str]
    _xaxis: Optional[str]
    _plottitle: Optional[str]
    _roottitle: Optional[str]
    _rootaxis: Optional[str]
    _mark: Optional[str]
    _paths: List[str]
    _plotmethod: Optional[str]

    def __init__(self, label: Union[Dict[str, str], str, Path, None]=None):
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

    def update(self, d: Dict[str,str]):
        for k, v in d.items():
            setattr(self, k, v)

    def format(self, *args, **kwargs):
        for name in (
                "text", "graph", "latex",
                "xaxis", "plottitle", "roottitle",
                "rootaxis"
                ):
            name = f"_{name}"
            oldvalue = getattr(self, name)
            if isinstance(oldvalue, str):
                setattr(self, name, oldvalue.format(*args, **kwargs))

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def index_values(self) -> List[str]:
        return self._index_values

    @property
    def index_dict(self) -> Dict[str,Tuple[str,int]]:
        return self._index_dict

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def text(self) -> Optional[str]:
        return self._text

    @text.setter
    def text(self, value: str):
        self._text = value

    @property
    def graph(self) -> Optional[str]:
        return self._graph or self._text

    @graph.setter
    def graph(self, value: Optional[str]):
        self._graph = value

    @property
    def latex(self) -> Optional[str]:
        return self._latex

    @latex.setter
    def latex(self, value: str):
        self._latex = value

    @property
    def plottitle(self) -> Optional[str]:
        return self._plottitle or self._latex or self._text

    @plottitle.setter
    def plottitle(self, value: Optional[str]):
        self._plottitle = value

    @property
    def roottitle(self) -> Optional[str]:
        if self._roottitle is not None:
            return self._roottitle
        return _latex_to_root(self.plottitle)

    @roottitle.setter
    def roottitle(self, value: Optional[str]):
        self._roottitle = value

    @property
    def rootaxis(self) -> Optional[str]:
        if self._rootaxis is not None:
            return self._rootaxis
        return _latex_to_root(self.axis)

    @rootaxis.setter
    def rootaxis(self, value: Optional[str]):
        self._rootaxis = value

    @property
    def axis(self) -> Optional[str]:
        return self._axis or self.plottitle

    @axis.setter
    def axis(self, value: Optional[str]):
        self._axis = value

    @property
    def xaxis(self) -> Optional[str]:
        return self._xaxis

    @xaxis.setter
    def xaxis(self, value: Optional[str]):
        self._xaxis = value

    @property
    def mark(self) -> Optional[str]:
        return self._mark

    @mark.setter
    def mark(self, value: str):
        self._mark = value

    @property
    def paths(self) -> List[str]:
        return self._paths

    @paths.setter
    def paths(self, value: List[str]):
        self._paths = value

    @property
    def plottable(self) -> bool:
        return self._plotmethod!="none"

    @property
    def plotmethod(self) -> Optional[str]:
        return self._plotmethod

    @plotmethod.setter
    def plotmethod(self, value: str):
        self._plotmethod = value.lower()

    def items(self):
        for k in self.__slots__:
            yield k, getattr(self, k)

    def __getitem__(self, k: str) -> Optional[str]:
        return getattr(self, k)

    def __setitem__(self, k: str, v: str) -> Optional[str]:
        return setattr(self, k, v)

    def get(self, k: str, default: str) -> Optional[str]:
        return getattr(self, k, default)

    def setdefault(self, k: str, default: str) -> Optional[str]:
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

    def inherit(self, source: "Labels", fmtlong: Union[str, Callable], fmtshort: Union[str, Callable]):
        fmtlong = _make_formatter(fmtlong)
        fmtshort = _make_formatter(fmtshort)

        inherit = ("_text", "_graph", "_latex", "_mark", "_axis", "_plottitle")
        kshort = {"_mark"}
        for _key in inherit:
            label = getattr(source, _key, None)
            if label is None: continue
            if isinstance(label, str):
                newv = fmtshort(label) if _key in kshort else fmtlong(label)
                if newv is not None:
                    self[_key] = newv
            else:
                self[_key] = label

def inherit_labels(
        source: dict,
        destination: Optional[dict]=None,
        *,
        fmtlong: Union[str, Callable],
        fmtshort: Union[str, Callable]
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

def _latex_to_root(text: Optional[str]) -> Optional[str]:
    if not text:
        return text
    return text.replace(r"\rm ", "").replace("\\", "#").replace("$","")
