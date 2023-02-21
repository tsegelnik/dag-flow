from .classwrapper import ClassWrapper
from .visitor import MakeDictWrapperVisitor
from .dictwrapperaccess import DictWrapperAccess
from collections.abc import Iterable, MutableMapping
import inspect

class DictWrapper(ClassWrapper):
    """Dictionary wrapper managing nested dictionaries
        The following functionality is implemented:
        - Tuple keys are treated to access nested dictionaries ('key1', 'key2', 'key3')
        - Optionally sep symbol may be set to automatically split string keys into tuple keys:
          'key1.key2.key3' will be treated as a nested key if '.' is set for the sep symbol
        - self._ may be used to access nested dictionaries via attributes: dw.key1.key2.key3
    """
    _sep: str = None
    _parent = None
    _type = dict
    def __new__(cls, dic, *args, **kwargs):
        if not isinstance(dic, (MutableMapping, DictWrapper)):
            return dic
        return ClassWrapper.__new__(cls)

    def __init__(self, dic, *, sep:str=None, parent=None):
        if isinstance(dic, DictWrapper):
            if sep is None:
                sep = dic._sep
            dic = dic._obj
        self._sep = sep
        self._type = type(dic)
        ClassWrapper.__init__(self, dic, types=MutableMapping)
        if parent:
            if sep and sep!=parent._sep:
                raise ValueError(f'Inconsistent separators: {sep} (self) and {parent._sep} (parent)')

            self._parent = parent
            self._sep = parent._sep
            self._types = parent._types

    @property
    def _(self):
        return DictWrapperAccess(self)

    def parent(self):
        return self._parent

    def child(self, key):
        try:
            ret = self[key]
        except KeyError:
            ret = self[key]=self._type()
            return self._wrap(ret)

        if not isinstance(ret, self._wrapper_class):
            raise KeyError('Child {!s} is not DictWrapper'.format(key))

        return ret

    def keys(self):
        return self._obj.keys()

    def iterkey(self, key):
        if isinstance(key, str):
            if self._sep:
                for s in key.split(self._sep):
                    yield s
            else:
                yield key
        elif isinstance(key, Iterable):
            for sk in key:
                for ssk in self.iterkey(sk):
                        yield ssk
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
            return self._wrap(self._obj.get(key, *args, **kwargs))

        sub = self._wrap(self._obj.get(key))
        if sub is None:
            if args:
                return args[0]
            raise KeyError( "No nested key '%s'"%key )
        return sub.get(rest, *args, **kwargs)

    def __getitem__(self, key):
        if key==():
            return self
        key, rest=self.splitkey(key)

        sub = self._wrap(self._obj.__getitem__(key))
        if not rest:
            return sub

        if sub is None:
            raise KeyError( "No nested key '%s'"%key )

        return sub[rest]

    def __delitem__(self, key):
        if key==():
            raise ValueError('May not delete itself')
        key, rest=self.splitkey(key)

        sub = self._wrap(self._obj.__getitem__(key))
        if not rest:
            del self._obj[key]
            return

        del sub[rest]

    def setdefault(self, key, value):
        key, rest=self.splitkey(key)

        if not rest:
            ret=self._obj.setdefault(key, value)
            return self._wrap(ret)

        if key in self:
            sub = self._wrap(self._obj.get(key))
        else:
            sub = self._obj[key] = self._type()
            sub = self._wrap(sub)
            # # cfg._set_parent( self )

        return sub.setdefault(rest, value)

    def set(self, key, value):
        key, rest=self.splitkey(key)

        if not rest:
            self._obj[key] = value
            return value

        if key in self:
            sub = self._wrap(self._obj.get(key))
        else:
            sub = self._obj[key] = self._type()
            sub = self._wrap(sub)
            # # cfg._set_parent( self )

        return sub.set(rest, value)

    __setitem__= set

    def __contains__(self, key):
        if key==():
            return True
        key, rest=self.splitkey(key)

        if not key in self._obj:
            return False

        if rest:
            sub = self._wrap(self._obj.get(key))
            return rest in sub

        return True

    def keys(self):
        return self._obj.keys()

    def items(self):
        for k, v in self._obj.items():
            yield k, self._wrap(v)

    def values(self):
        for v in self._obj.values():
            yield self._wrap(v)

    def deepcopy(self):
        new = DictWrapper(self._type())
        for k, v in self.items():
            k = k,
            if isinstance(v, self._wrapper_class):
                new[k] = v.deepcopy()._obj
            else:
                new[k] = v

        new._sep = self._sep

        return new

    def walkitems(self, startfromkey=(), *, appendstartkey=False, maxdepth=None):
        v0 = self[startfromkey]
        k0 = tuple(self.iterkey(startfromkey))
        startdepth=len(k0)

        if maxdepth is None:
            nextdepth=None
        else:
            maxdepth-=startdepth

            nextdepth=maxdepth-1
            if nextdepth<0:
                nextdepth=0

        if maxdepth==0 or not isinstance(v0, self._wrapper_class):
            if appendstartkey:
                yield k0, v0
            else:
                yield (), v0
            return

        if not appendstartkey:
            k0 = ()

        for k, v in v0.items():
            k = k,
            if isinstance(v, self._wrapper_class):
                for k1, v1 in v.walkitems(maxdepth=nextdepth):
                    yield k0+k+k1, v1
            else:
                yield k0+k, v

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
        for k, v in self.walkitems(*args, **kwargs):
            yield k

    def walkvalues(self, *args, **kwargs):
        for k, v in self.walkitems(*args, **kwargs):
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
