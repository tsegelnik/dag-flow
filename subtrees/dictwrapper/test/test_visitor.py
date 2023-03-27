from multikeydict.nestedmkdict import NestedMKDict
from multikeydict.visitor import NestedMKDictVisitorDemostrator

def test_nestedmkdict_04_visitor():
    dct = dict([('a', 1), ('b', 2), ('c', 3), ('d', dict(e=4)), ('f', dict(g=dict(h=5)))])
    dct['z.z.z'] = 0
    dw = NestedMKDict(dct)

    keys0 = (('a',) , ('b',) , ('c',) , ('d', 'e'), ('f', 'g', 'h'), ('z.z.z', ))
    values0 = (1, 2, 3, 4, 5, 0)

    keys = tuple(dw.walkkeys())
    values = tuple(dw.walkvalues())
    assert keys==keys0
    assert values==values0

    class Visitor(object):
        keys, values = (), ()
        def __call__(self, k, v):
            self.keys+=k,
            self.values+=v,
    v = Visitor()
    dw.visit(v)
    assert v.keys==keys0
    assert v.values==values0

def test_nestedmkdict_05_visitor():
    dct = dict([('a', 1), ('b', 2), ('c', 3), ('d', dict(e=4)), ('f', dict(g=dict(h=5)))])
    dct['z.z.z'] = 0
    dw = NestedMKDict(dct)

    dw.visit(NestedMKDictVisitorDemostrator())
