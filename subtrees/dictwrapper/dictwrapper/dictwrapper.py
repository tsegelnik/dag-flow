from .classwrapper import ClassWrapper
from .visitor import MakeDictWrapperVisitor
from .dictwrapperaccess import DictWrapperAccess

from collections.abc import Sequence, MutableMapping
from typing import Any

class DictWrapper(ClassWrapper):
    """Dictionary wrapper managing nested dictionaries
        The following functionality is implemented:
        - Tuple keys are treated to access nested dictionaries ('key1', 'key2', 'key3')
        - Optionally sep symbol may be set to automatically split string keys into tuple keys:
          'key1.key2.key3' will be treated as a nested key if '.' is set for the sep symbol
        - self._ may be used to access nested dictionaries via attributes: dw.key1.key2.key3
    """
    _sep: str = None
    _parent: Any = None
    _types: Any = dict
    _not_recursive_to_others: bool = True
    def __new__(cls, dic, *args, parent=None, sep=None, recursive_to_others=None):
        if not isinstance(dic, (MutableMapping, DictWrapper)):
            return dic
        return ClassWrapper.__new__(cls)

    def __init__(self, dic, *, sep: str=None, parent=None, recursive_to_others: bool=False):
        if isinstance(dic, DictWrapper):
            if sep is None:
                sep = dic._sep
            recursive_to_others = not dic._not_recursive_to_others
            dic = dic._object
        super().__init__(dic, types=type(dic))

        self._sep = sep
        self._not_recursive_to_others = not recursive_to_others
        if parent:
            if sep and sep!=parent._sep:
                raise ValueError(f'Inconsistent separators: {sep} (self) and {parent._sep} (parent)')

            self._parent = parent
            self._sep = parent._sep
            self._types = parent._types
            self._not_recursive_to_others = parent._not_recursive_to_others

    @property
    def _(self):
        return DictWrapperAccess(self)

    def parent(self):
        return self._parent

    def child(self, key):
        try:
            ret = self[key]
        except KeyError:
            ret = self[key]=self._types()
            return self._wrap(ret, parent=self)

        if not isinstance(ret, self._wrapper_class):
            raise KeyError('Child {!s} is not DictWrapper'.format(key))

        return ret

    def keys(self):
        return self._object.keys()

    def iterkey(self, key):
        if isinstance(key, str):
            if self._sep:
                yield from key.split(self._sep)
            else:
                yield key
        elif isinstance(key, Sequence):
            for sk in key:
                yield from self.iterkey(sk)
        else:
            yield key

    def splitkey(self, key):
        it = self.iterkey(key)
        try:
            return next(it), tuple(it)
        except StopIteration:
            return None, None

    def get(self, key, *args, **kwargs):
        if key==():
            return self
        key, rest=self.splitkey(key)

        if not rest:
            ret = self._object.get(key, *args, **kwargs)
            return self._wrap(ret, parent=self)

        sub = self._wrap(self._object.get(key), parent=self)
        if sub is None:
            if args:
                return args[0]
            raise KeyError(f"No nested key '{key}'")

        if self._not_recursive_to_others and not isinstance(sub, DictWrapper):
            raise TypeError(f"Nested value for '{key}' has wrong type")

        return sub.get(rest, *args, **kwargs)

    def __getitem__(self, key):
        if key==():
            return self
        key, rest=self.splitkey(key)

        sub = self._object.__getitem__(key)
        sub = self._wrap(sub, parent=self)
        if not rest:
            return sub

        if sub is None:
            raise KeyError( f"No nested key '{key}'" )

        if self._not_recursive_to_others and not isinstance(sub, DictWrapper):
            raise TypeError(f"Nested value for '{key}' has wrong type")

        return sub[rest]

    def __delitem__(self, key):
        if key==():
            raise ValueError('May not delete itself')
        key, rest=self.splitkey(key)

        sub = self._wrap(self._object.__getitem__(key), parent=self)
        if not rest:
            del self._object[key]
            return

        if self._not_recursive_to_others and not isinstance(sub, DictWrapper):
            raise TypeError(f"Nested value for '{key}' has wrong type")

        del sub[rest]

    def setdefault(self, key, value):
        key, rest=self.splitkey(key)

        if not rest:
            ret=self._object.setdefault(key, value)
            return self._wrap(ret, parent=self)

        if key in self:
            sub = self._wrap(self._object.get(key), parent=self)
        else:
            sub = self._object[key] = self._types()
            sub = self._wrap(sub, parent=self)
            # # cfg._set_parent( self )

        if self._not_recursive_to_others and not isinstance(sub, DictWrapper):
            raise TypeError(f"Nested value for '{key}' has wrong type")

        return sub.setdefault(rest, value)

    def set(self, key, value):
        key, rest=self.splitkey(key)

        if not rest:
            self._object[key] = value
            return value

        if key in self:
            sub = self._wrap(self._object.get(key), parent=self)
        else:
            sub = self._object[key] = self._types()
            sub = self._wrap(sub, parent=self)
            # # cfg._set_parent( self )

        if self._not_recursive_to_others and not isinstance(sub, DictWrapper):
            raise TypeError(f"Nested value for '{key}' has wrong type")

        return sub.set(rest, value)

    __setitem__= set

    def __contains__(self, key):
        if key==():
            return True
        key, rest=self.splitkey(key)

        if key not in self._object:
            return False

        if rest:
            sub = self._wrap(self._object.get(key), parent=self)

            if self._not_recursive_to_others and not isinstance(sub, DictWrapper):
                raise TypeError(f"Nested value for '{key}' has wrong type")

            return rest in sub

        return True

    def keys(self):
        return self._object.keys()

    def items(self):
        for k, v in self._object.items():
            yield k, self._wrap(v, parent=self)

    def values(self):
        for v in self._object.values():
            yield self._wrap(v, parent=self)

    def deepcopy(self):
        new = DictWrapper(self._types(), parent=self._parent, sep=self._sep, recursive_to_others=not self._not_recursive_to_others)
        for k, v in self.items():
            k = k,
            if isinstance(v, self._wrapper_class):
                new[k] = v.deepcopy()._object
            else:
                new[k] = v

        new._sep = self._sep

        return new

    def walkitems(self, startfromkey=(), *, appendstartkey=False, maxdepth=None):
        v0 = self[startfromkey]
        k0 = tuple(self.iterkey(startfromkey))

        if maxdepth is None:
            nextdepth=None
        else:
            nextdepth=max(maxdepth-len(k0)-1, 0)

        if maxdepth==0 or not isinstance(v0, self._wrapper_class):
            if appendstartkey:
                yield k0, v0
            else:
                yield (), v0
            return

        if not appendstartkey:
            k0 = ()

        for k, v in v0.items():
            k = k0+(k,)
            if isinstance(v, self._wrapper_class):
                for k1, v1 in v.walkitems(maxdepth=nextdepth):
                    yield k+k1, v1
            elif not self._not_recursive_to_others and isinstance(v, MutableMapping):
                for k1, v1 in v.items():
                    if isinstance(k1, tuple):
                        yield k+k1, v1
                    else:
                        yield k+(k1,), v1
            else:
                yield k, v

    def walkdicts(self):
        yieldself=True
        for k, v in self.items():
            k = k,
            if isinstance(v, self._wrapper_class):
                yieldself=False
                for k1, v1 in v.walkdicts():
                    yield k+k1, v1
        if yieldself:
            yield (), self

    def walkkeys(self, *args, **kwargs):
        for k, _ in self.walkitems(*args, **kwargs):
            yield k

    def walkvalues(self, *args, **kwargs):
        for _, v in self.walkitems(*args, **kwargs):
            yield v

    def visit(self, visitor, parentkey=()):
        visitor = MakeDictWrapperVisitor(visitor)

        if not parentkey:
            visitor.start(self)

        visitor.enterdict(parentkey, self)
        for k, v in self.items():
            key = parentkey + (k,)
            if isinstance(v, self._wrapper_class):
                v.visit(visitor, parentkey=key)
            else:
                visitor.visit(key, v)

        visitor.exitdict(parentkey, self)

        if not parentkey:
            visitor.stop(self)
