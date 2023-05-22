from typing import Optional, Union, Callable, Dict, Tuple

def repr_pretty(self, p, cycle):
    """Pretty repr for IPython. To be used as __repr__ method"""
    p.text(str(self) if not cycle else '...')

def _make_formatter(fmt: Union[str, Callable, dict]) -> Callable:
    if isinstance(fmt, str):
        return fmt.format
    elif isinstance(fmt, dict):
        return lambda s: fmt.get(s, s)

    return fmt

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

    kshort = {'mark'}
    kskip = {'key', 'name'}
    for k, v in source.items():
        if k in kskip:
            continue
        newv = fmtshort(v) if k in kshort else fmtlong(v)
        if newv is not None:
            destination[k] = newv

    return destination

class Labels:
    __slots__ = ('name', 'text', '_graph', 'latex', 'mark', '_axis', '_plottitle', 'key', 'paths')

    name: Optional[str]
    text: Optional[str]
    _graph: Optional[str]
    latex: Optional[str]
    _axis: Optional[str]
    _plottitle: Optional[str]
    mark: Optional[str]
    key: Optional[str]
    paths: Tuple[str]

    def __init__(self, label: Union[Dict[str, str], str, None]=None):
        for slot in self.__slots__:
            setattr(self, slot, None)
        self.paths = []

        if isinstance(label, str):
            self.text = label
        elif isinstance(label, dict):
            for k, v in label.items():
                setattr(self, k, v)

    @property
    def graph(self) -> Optional[str]:
        return self._graph or self.text

    @graph.setter
    def graph(self, value: Optional[str]):
        self._graph = value

    @property
    def plottitle(self) -> Optional[str]:
        return self._plottitle or self.latex or self.text

    @plottitle.setter
    def plottitle(self, value: Optional[str]):
        self._plottitle = value

    @property
    def axis(self) -> Optional[str]:
        return self._axis or self.plottitle

    @axis.setter
    def axis(self, value: Optional[str]):
        self._axis = value

    def setdefaults(self, labels: dict):
        for k, v in labels.items():
            if getattr(self, k) is None:
                setattr(self, k, v)

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
