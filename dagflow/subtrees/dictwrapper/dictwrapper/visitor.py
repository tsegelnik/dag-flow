class DictWrapperVisitor(object):
    def start(self, dct):
        pass

    def enterdict(self, k, v):
        pass

    def visit(self, k, v):
        pass

    def exitdict(self, k, v):
        pass

    def stop(self, dct):
        pass

def MakeDictWrapperVisitor(fcn):
    if isinstance(fcn, DictWrapperVisitor):
        return fcn

    if not callable(fcn):
        raise TypeError(f'Expect function, got {type(fcn).__name__}')

    ret=DictWrapperVisitor()
    ret.visit = fcn
    return ret

class DictWrapperVisitorDemostrator(DictWrapperVisitor):
    fmt = '{action:7s} {depth!s:>5s} {key!s:<{keylen}s} {vtype!s:<{typelen}s} {value}'
    opts = dict(keylen=20, typelen=15)
    def typestring(self, v):
        return type(v).__name__

    def start(self, d):
        v = object.__repr__(d.object)
        print('Start printing dictionary:', v)
        self._print('Action', 'Key', 'Value', 'Type', depth='Depth')

    def stop(self, _):
        print('Done printing dictionary')

    def enterdict(self, k, d):
        d = d.object
        v = object.__repr__(d)
        self._print('Enter', k, v, self.typestring(d))

    def exitdict(self, k, d):
        d = d.object
        v = object.__repr__(d)
        self._print('Exit', k, v, self.typestring(d))

    def visit(self, k, v):
        self._print('Visit', k, v, self.typestring(v))

    def _print(self, action, k, v, vtype, *, depth=None):
        if depth is None:
            depth = len(k)
        print(self.fmt.format(action=action, depth=depth, key=k, vtype=vtype, value=v, **self.opts))

