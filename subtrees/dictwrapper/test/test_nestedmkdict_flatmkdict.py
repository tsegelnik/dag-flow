from multikeydict.nestedmkdict import NestedMKDict
from multikeydict.flatmkdict import FlatMKDict
from pytest import raises

def test_nestedmkdict_flatmkdict():
    flatmkdict = FlatMKDict({
        ('a1', 'b1', 'c1'): 'v1',
        ('a2', 'b2', 'c2'): 'v2',
        })
    dct = {'root': {
        'subfolder1': {
            'key1': 'value1',
            'key2': 'value2'
        },
        'subfolder2': {
            'key1': 'value1',
            'key2': 'value2',
            'st': flatmkdict
        },
        'key0': 'value0'
    }}
    dw = NestedMKDict(dct, recursive_to_others=True)
    dws = NestedMKDict(dct, sep='.', recursive_to_others=True)
    dwerror = NestedMKDict(dct, recursive_to_others=False)

    objects = (dw, dws, dwerror)
    objectsok = (dw, dws)

    assert flatmkdict['a1', 'b1', 'c1']=='v1'
    assert flatmkdict['b1', 'a1', 'c1']=='v1'
    assert flatmkdict['c1', 'a1', 'b1']=='v1'

    for obj in objects:
        st1 = obj['root', 'subfolder2', 'st']
        assert st1 is flatmkdict

    for obj in objectsok:
        assert obj['root', 'subfolder2', 'st', 'a1', 'b1', 'c1']=='v1'
        assert obj['root', 'subfolder2', 'st', 'b1', 'a1', 'c1']=='v1'
        assert obj['root', 'subfolder2', 'st', 'c1', 'a1', 'b1']=='v1'
        assert obj['root', 'subfolder2', 'st', 'a2', 'b2', 'c2']=='v2'
        assert obj['root', 'subfolder2', 'st', 'b2', 'a2', 'c2']=='v2'
        assert obj['root', 'subfolder2', 'st', 'c2', 'a2', 'b2']=='v2'

        assert ('root', 'subfolder2', 'st', 'a1', 'b1', 'c1') in obj
        assert ('root', 'subfolder2', 'st', 'b1', 'a1', 'c1') in obj
        assert ('root', 'subfolder2', 'st', 'c1', 'a1', 'b1') in obj
        assert ('root', 'subfolder2', 'st', 'a2', 'b2', 'c2') in obj
        assert ('root', 'subfolder2', 'st', 'b2', 'a2', 'c2') in obj
        assert ('root', 'subfolder2', 'st', 'c2', 'a2', 'b2') in obj

    assert dws['root.subfolder2.st.a1.b1.c1']=='v1'
    assert dws['root.subfolder2.st.b1.a1.c1']=='v1'
    assert dws['root.subfolder2.st.c1.a1.b1']=='v1'
    assert dws['root.subfolder2.st.a2.b2.c2']=='v2'
    assert dws['root.subfolder2.st.b2.a2.c2']=='v2'
    assert dws['root.subfolder2.st.c2.a2.b2']=='v2'

    assert 'root.subfolder2.st.a1.b1.c1' in dws
    assert 'root.subfolder2.st.b1.a1.c1' in dws
    assert 'root.subfolder2.st.c1.a1.b1' in dws
    assert 'root.subfolder2.st.a2.b2.c2' in dws
    assert 'root.subfolder2.st.b2.a2.c2' in dws
    assert 'root.subfolder2.st.c2.a2.b2' in dws

    assert 'root.subfolder3.st.c2.a2.b2' not in dws
    assert 'root.subfolder2.st.c3.a2.b2' not in dws

    with raises(KeyError):
        dws['root.subfolder2.st.a1.b2.c1']

    with raises(KeyError):
        dws['root.subfolder1.st.a1.b1.c1']

    with raises(TypeError):
        dwerror['root', 'subfolder2', 'st', 'a1', 'b1', 'c1']

    with raises(TypeError):
        dwerror.get(('root', 'subfolder2', 'st', 'a1', 'b1', 'c1'))

    with raises(TypeError):
        dwerror.get(('root', 'subfolder2', 'st', 'a1', 'b1', 'c1'), 'default')

    with raises(TypeError):
        del dwerror['root', 'subfolder2', 'st', 'a1', 'b1', 'c1']

    with raises(TypeError):
        dwerror.setdefault(('root', 'subfolder2', 'st', 'a1', 'b1', 'c1'), 'default')

    with raises(TypeError):
        dwerror.set(('root', 'subfolder2', 'st', 'a1', 'b1', 'c1'), 'default')

    with raises(TypeError):
        ('root', 'subfolder2', 'st', 'a1', 'b1', 'c1') in dwerror

    # Walks
    for k, v in dw.walkitems():
        print(k, v)

