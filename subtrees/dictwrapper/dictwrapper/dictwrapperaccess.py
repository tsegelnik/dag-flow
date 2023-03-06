class DictWrapperAccess(object):
    '''DictWrapper wrapper. Enables attribute based access to nested dictionaries'''
    _ = None
    def __init__(self, dct):
        self.__dict__['_'] = dct

    def __call__(self, key):
        return self._.child(key)._

    def __getattr__(self, key):
        ret = self._[key]

        if isinstance(ret, self._._wrapper_class):
            return ret._

        return ret

    def __setattr__(self, key, value):
        self._[key]=value

    def __delattr__(self, key):
        del self._[key]

    def __dir__(self):
        return list(self._.keys())
