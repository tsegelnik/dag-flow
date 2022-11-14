
from collections import OrderedDict
from collections.abc import Iterable

from .tools import IsIterable, nth


class EdgeContainer:
    _dict = None
    _dtype = None

    def __init__(self, iterable=None):
        self._dict = OrderedDict()
        if iterable:
            self.__iadd__(iterable)

    def __iadd__(self, value):
        if IsIterable(value):
            for v in value:
                self.__iadd__(v)
            return self
        if self._dtype and not isinstance(value, self._dtype):
            raise RuntimeError(
                f"The type {type(value)} of the data doesn't correpond "
                f"to {self._dtype}!"
            )
        name = value.name
        if not name:
            raise RuntimeError("May not add objects with undefined name")
        if name in self._dict:
            raise RuntimeError("May not add duplicated items")
        self._dict[name] = value
        return self

    def __getitem__(self, key):
        if isinstance(key, int):
            return nth(self._dict.values(), key)
        elif isinstance(key, str):
            return self._dict[key]
        elif isinstance(key, slice):
            return tuple(self._dict.values())[key]
        elif isinstance(key, Iterable):
            return tuple(self.__getitem__(k) for k in key)
        raise TypeError(f"Unsupported key type: {type(key).__name__}")

    def __getattr__(self, name):
        if name.startswith('__'):
            return super().__getattr__(name)

        return self._dict[name]

    def __len__(self):
        return len(self._dict)

    def __dir__(self):
        return self._dict.keys()

    def __iter__(self):
        return iter(self._dict.values())

    def __contains__(self, name):
        return name in self._dict
