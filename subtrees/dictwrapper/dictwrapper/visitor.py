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
        print(self.fmt.format(action='Action', depth='Depth', key='Key', vtype='Type', value='Value', **self.opts))

    def stop(self, d):
        print('Done printing dictionary')

    def enterdict(self, k, d):
        d = d.object
        v = object.__repr__(d)
        print(self.fmt.format(action='Enter', depth=len(k), key=k, vtype=self.typestring(d), value=v, **self.opts))

    def exitdict(self, k, d):
        d = d.object
        v = object.__repr__(d)
        print(self.fmt.format(action='Exit', depth=len(k), key=k, vtype=self.typestring(d), value=v, **self.opts))

    def visit(self, k, v):
        print(self.fmt.format(action='Visit', depth=len(k), key=k, vtype=self.typestring(v), value=v, **self.opts))

