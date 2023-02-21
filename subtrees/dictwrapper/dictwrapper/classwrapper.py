class ClassWrapper(object):
    def __init__(self, obj, *, parent=None, types=None):
        self._obj = obj
        if types:
            self._types = types
        else:
            self._types = type(obj)
        self._wrapper_class = type(self)

    def unwrap(self):
        return self._obj

    def __str__(self):
        return str(self._obj)

    def __repr__(self):
        return repr(self._obj)

    def __dir__(self):
        return dir(self._obj)

    def __len__(self):
        return len(self._obj)

    def __bool__(self):
        return bool(self._obj)

    def __contains__(self, v):
        return v in self._obj

    def __eq__(self, other):
        if isinstance(other, ClassWrapper):
            return self._obj==other._obj

        return self._obj==other

    def _wrap(self, obj):
        if isinstance(obj, ClassWrapper):
            return obj

        if isinstance(obj, self._types):
            return self._wrapper_class(obj, parent=self)

        return obj

