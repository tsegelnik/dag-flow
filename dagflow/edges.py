from collections.abc import Sequence

from .exception import CriticalError
from .iter import IsIterable

from typing import List, Dict, Union

class EdgeContainer:
    _kw_edges: Dict
    _pos_edges: List
    _all_edges: Dict
    _dtype = None

    def __init__(self, iterable=None):
        self._kw_edges = {}
        self._pos_edges = []
        self._all_edges = {}
        if iterable:
            self.add(iterable)

    def add(self, value, *, positional: bool=True, keyword: bool=True):
        if positional==keyword==False:
            raise RuntimeError('Edge should be at least positional or a keyword')

        if IsIterable(value):
            for v in value:
                self.add(v, positional=positional, keyword=keyword)
            return self
        if self._dtype and not isinstance(value, self._dtype):
            raise RuntimeError(
                f"The type {type(value)} of the data doesn't correpond "
                f"to {self._dtype}!"
            )
        name = value.name
        if not name:
            raise RuntimeError("May not add objects with undefined name")
        if name in self._all_edges:
            raise RuntimeError("May not add duplicated items")

        if positional:
            self._pos_edges.append(value)
        if keyword:
            self._kw_edges[name] = value
        self._all_edges[name]=value
        return self

    def allocate(self) -> bool:
        return all(edge.allocate() for edge in self._all_edges.values())

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._kw_edges[key]
        elif isinstance(key, (int, slice)):
            return self._pos_edges[key]
        elif isinstance(key, Sequence):
            return tuple(self.__getitem__(k) for k in key)
        raise TypeError(f"Unsupported key type: {type(key).__name__}")

    def get(self, key, default = None):
        try:
            return self.__getitem__(key)
        except Exception:
            return default

    def has_key(self, key: str) -> bool:
        return key in self._kw_edges

    def get_pos(self, idx: int):
        """Get positional leg"""
        return self._pos_edges[idx]
    iat = get_pos

    def index(self, arg):
        return self._pos_edges.index(arg)

    def get_kw(self, key: str):
        """Return keyword leg"""
        return self._kw_edges[key]
    kat = get_kw

    def len_pos(self) -> int:
        """Returns a number of the positional legs"""
        return len(self._pos_edges)
    __len__ = len_pos

    def len_kw(self) -> int:
        """Returns a number of the keyword legs"""
        return len(self._kw_edges)

    def len_all(self) -> int:
        """Returns a number of the all legs"""
        return len(self._all_edges)

    def __iter__(self):
        return iter(self._pos_edges)

    def iter_all(self):
        return iter(self._all_edges.values())

    def iter_data(self):
        for edge in self._pos_edges:
            yield edge.data

    def iter(self, key: Union[int, str, slice, Sequence]):
        if isinstance(key, int):
            yield self._pos_edges[key]
        elif isinstance(key, str):
            yield self._kw_edges[key]
        elif isinstance(key, slice):
            yield from self._pos_edges[key]
        elif isinstance(key, Sequence):
            for subkey in key:
                if isinstance(subkey, int):
                    yield self._pos_edges[subkey]
                elif isinstance(subkey, str):
                    yield self._kw_edges[subkey]
                elif isinstance(subkey, slice):
                    yield from self._pos_edges[subkey]
                else:
                    raise CriticalError(f'Invalid subkey type {type(subkey).__name__}')
        else:
            raise CriticalError(f'Invalid key type {type(key).__name__}')

    def __contains__(self, name):
        return name in self._all_edges

    def _replace(self, old, new):
        replaced = False

        for k, v in self._kw_edges.items():
            if old is v:
                self._kw_edges[k] = new
                replaced = True

        for i, v in enumerate(self._pos_edges):
            if old is v:
                self._pos_edges[i] = new
                replaced = True

        if not replaced:
            raise CriticalError('Unable to replace an output/input (not found)')
