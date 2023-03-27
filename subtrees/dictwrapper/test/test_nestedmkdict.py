from multikeydict.nestedmkdict import NestedMKDict
import pytest

def test_nestedmkdict_01():
    dw = NestedMKDict({})

    assert not dw
    assert len(dw)==0

def test_nestedmkdict_02():
    dw = NestedMKDict(dict(a=1))

    assert dw
    assert len(dw)==1

def test_nestedmkdict_03():
    d = dict(a=1, b=2, c=3)
    dw = NestedMKDict(d)

    assert dw.get('a')==1
    assert dw.get('b')==2
    assert dw.get('c')==3
    assert dw.get('d')==None
    assert dw.get('d.e')==None

    assert tuple(dw.keys())==('a','b','c')

@pytest.mark.parametrize('sep', [None, '.'])
def test_nestedmkdict_03(sep):
    dct = dict(a=1, b=2, c=3, d=dict(e=4), f=dict(g=dict(h=5)))
    dct['z.z.z'] = 0
    print(dct)
    dw = NestedMKDict(dct, sep=sep)

    #
    # Test self access
    #
    assert dw.get(()).object is dct
    assert dw[()].object is dct

    #
    # Test wrapping
    #
    assert isinstance(dw.get('d'), NestedMKDict)
    assert isinstance(dw.get(('f', 'g')), NestedMKDict)

    #
    # Test get tuple
    #
    assert dw.get(('d', 'e'))==4
    assert dw.get(('d', 'e1')) is None
    assert dw.get(('f', 'g', 'h'))==5
    try:
        dw.get(('z', 'z', 'z'))
        assert False
    except KeyError:
        pass

    #
    # Test getitem tuple
    #
    assert dw[('d', 'e')]==4
    try:
        dw[('d', 'e1')]
        assert False
    except KeyError:
        pass
    assert dw[('f', 'g', 'h')]==5

    try:
        dw[('z', 'z', 'z')]
        assert False
    except KeyError:
        pass

    #
    # Test get sep
    #
    if sep:
        assert dw.get('d.e')==4
    else:
        assert dw.get('d.e') is None

    if sep:
        try:
            dw.get('z.z.z')
            assert False
        except KeyError:
            pass
    else:
        assert dw.get('z.z.z')==0

    #
    # Test getitem sep
    #
    try:
        assert dw['d.e']==4
        assert sep is not None
    except KeyError:
        pass

    try:
        assert dw['f.g.h']==5
        assert dw[('f.g', 'h')]==5
        assert sep is not None
    except KeyError:
        pass

    if sep:
        try:
            dw['z.z.z']
            assert False
        except KeyError:
            pass
    else:
        assert dw['z.z.z']==0

    #
    # Test contains
    #
    assert 'a' in dw
    assert not 'a1' in dw
    assert 'd' in dw

    #
    # Test contains tuple
    #
    assert ('d', 'e') in dw
    assert not ('k', 'e') in dw
    assert ('f', 'g', 'h') in dw
    assert ('f.g.h' in dw) == bool(sep)
    assert ('z.z.z' in dw) == bool(not sep)

    #
    # Test parents
    #
    g = dw.get(('f', 'g'))
    assert g.parent().parent() is dw

    #
    # Test children
    #
    m=dw.child(('k', 'l', 'm'))
    assert dw.get(('k', 'l', 'm')).object is m.object

    #
    # Test recursive setitem
    #
    dw[('k', 'l', 'm', 'n')] = 5
    try:
        dw.child(tuple('klmn'))
        assert False
    except KeyError:
        pass
    assert dw.get(('k', 'l', 'm', 'n')) == 5

    dw[('o.l.m.n')] = 6
    assert dw['o.l.m.n'] == 6
    if not sep:
        assert dw.object['o.l.m.n'] == 6

    #
    # Test attribute access
    #
    assert dw._.a==1
    assert dw._.b==2
    assert dw._.c==3
    assert dw._.d.e==4
    assert dw._.f.g.h==5

    dw._.f.g.h=6
    assert dw._.f.g.h==6
    assert dw._._ is dw

def test_nestedmkdict_06_inheritance():
    dct = dict([('a', 1), ('b', 2), ('c', 3), ('d', dict(e=4)), ('f', dict(g=dict(h=5, i=6)))])
    dct['z.z.z'] = 0

    class NestedMKDictA(NestedMKDict):
        def count(self):
            return len(tuple(self.walkitems()))

        def depth(self):
            return max([len(k) for k in self.walkkeys()])

    dw = NestedMKDictA(dct, sep='.')
    assert dw.count()==7
    assert dw['d'].count()==1
    assert dw['f'].count()==2
    assert dw['f.g'].count()==2
    assert dw._.f._.count()==2

    assert dw.depth()==3
    assert dw['d'].depth()==1
    assert dw['f'].depth()==2

def test_nestedmkdict_07_delete():
    dct = dict([('a', 1), ('b', 2), ('c', 3), ('d', dict(e=4)), ('f', dict(g=dict(h=5)))])
    dct['z.z.z'] = 0
    dw = NestedMKDict(dct)

    assert 'a' in dw
    del dw['a']
    assert 'a' not in dw

    assert ('d', 'e') in dw
    del dw[('d', 'e')]
    assert ('d', 'e') not in dw

    assert ('f', 'g', 'h') in dw
    del dw._.f.g.h
    assert ('f', 'g', 'h') not in dw
    assert ('f', 'g') in dw

def test_nestedmkdict_08_create():
    dct = dict([('a', 1), ('b', 2), ('c', 3), ('d', dict(e=4)), ('f', dict(g=dict(h=5)))])
    dct['z.z.z'] = 0
    dw = NestedMKDict(dct, sep='.')

    dw._('i.k').l=3
    assert dw._.i.k.l==3

    child = dw.child('child')
    assert dw['child'].object=={}

def test_nestedmkdict_09_dictcopy():
    dct = dict([('a', 1), ('b', 2), ('c', 3), ('d', dict(e=4)), ('f', dict(g=dict(h=5)))])
    dct['z'] = {}
    dw = NestedMKDict(dct, sep='.')

    dw1 = dw.deepcopy()
    for i, (k, v) in enumerate(dw1.walkdicts()):
        # print(i, k)
        assert k in dw
        assert v._object==dw[k]._object
        assert v._object is not dw[k]._object
        assert type(v._object) is type(dw[k]._object)
    assert i==2

def test_nestedmkdict_09_walkitems():
    dct = {
        'a': 1,
        'b': 2,
        'c': 3,
        'c1': {
            'i': {
                'j': {
                    'k': {
                        'l': 6
                    }
                }
            }
        },
        'd': {
            'e': 4
        },
        'f': {
            'g': {
                'h': 5
            }
        }
    }
    dct['z'] = {}
    dw = NestedMKDict(dct, sep='.')

    imaxlist=[5, 0, 6, 5, 5, 5, 5, 5, 5]
    for imax, maxdepth in zip(imaxlist, [None]+list(range(len(imaxlist)))):
        i=0
        print(f'{imax=}, {maxdepth=}')
        maxk = -1
        for i, (k, v) in enumerate(dw.walkitems(maxdepth=maxdepth)):
            print(i, k, v)
            assert maxdepth is None or len(k)<=maxdepth
            maxk=max(maxk, len(k))
        print(f'{maxk=}')
        print()
        assert i==imax

def test_nestedmkdict_09_walk():
    dct = dict([('a', 1), ('b', 2), ('c', 3), ('d', dict(e=4)), ('f', dict(g=dict(h=5)))])
    dw = NestedMKDict(dct)

    keys0 = [ ('a',), ('b', ), ('c',), ('d', 'e'), ('f', 'g', 'h') ]
    keys = [k for k, v in dw.walkitems()]
    assert keys==keys0

    assert [(k,v) for k, v in dw.walkitems('a', appendstartkey=True)] == [(('a',), 1)]
    assert [(k,v) for k, v in dw.walkitems('a', appendstartkey=False)] == [((), 1)]
    assert [(k,v) for k, v in dw.walkitems('d', appendstartkey=True)] == [(('d','e'), 4)]
    assert [(k,v) for k, v in dw.walkitems('d', appendstartkey=False)] == [(('e',), 4)]
    assert [(k,v) for k, v in dw.walkitems(('f','g'), appendstartkey=True)] == [(('f','g', 'h'), 5)]
    assert [(k,v) for k, v in dw.walkitems(('f','g'), appendstartkey=False)] == [(('h',), 5)]

def test_nestedmkdict_10_iterkey():
    d = dict(a=1, b=2, c=3)
    dw = NestedMKDict(d)

    assert ['a']==list(dw.iterkey('a'))
    assert ['a.b']==list(dw.iterkey('a.b'))
    assert ['a', 'b']==list(dw.iterkey(('a', 'b')))
    assert [1]==list(dw.iterkey(1))
    assert [1.0]==list(dw.iterkey(1.0))

def test_nestedmkdict_11_iterkey():
    d = dict(a=1, b=2, c=3)
    dw = NestedMKDict(d,  sep='.')

    assert ['a']==list(dw.iterkey('a'))
    assert ['a', 'b']==list(dw.iterkey('a.b'))
    assert ['a', 'b']==list(dw.iterkey(('a', 'b')))
    assert [1]==list(dw.iterkey(1))
    assert [1.0]==list(dw.iterkey(1.0))

def test_nestedmkdict_setdefault_01():
    d = dict(a=dict(b=dict(key='value')))
    dw = NestedMKDict(d)

    newdict = dict(newkey='newvalue')

    sd1 = dw.setdefault(('a','b'), newdict)
    assert isinstance(sd1, NestedMKDict)
    assert sd1._object==d['a']['b']

    sd2 = dw.setdefault(('a','c'), newdict)
    assert isinstance(sd2, NestedMKDict)
    assert sd2._object==newdict

def test_nestedmkdict_eq_01():
    d = dict(a=dict(b=dict(key='value')))
    dw = NestedMKDict(d)

    assert dw['a']==d['a']
    assert d['a']==dw['a']
    assert dw['a']!=d
    assert dw['a']==dw['a']
    assert dw['a'] is not dw['a']
