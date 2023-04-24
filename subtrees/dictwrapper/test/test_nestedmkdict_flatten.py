from multikeydict.nestedmkdict import NestedMKDict
from multikeydict.flatten import flatten, _select
from pytest import raises

from pprint import pprint

def test__select():
    a, b = _select(tuple('abcd'), set('cd'))
    assert a==tuple('ab')
    assert b==tuple('cd')

    a, b = _select(tuple('abcd'), set('bd'))
    assert a==tuple('abc')
    assert b==tuple('d')

    a, b = _select(tuple('abcd'), set('ab'))
    assert a==tuple('abcd')
    assert b==tuple()

    a, b = _select(tuple('abcd'), set('ef'))
    assert a==tuple('abcd')
    assert b==tuple()


def test_nestedmkdict_flatten_v01():
    dct = {'root': {
        'subfolder1': {
            'key1': 'value1',
            'key2': 'value2'
        },
        'subfolder2': {
            'key1': 'value1',
            'key2': 'value2',
            'st': {
                'a1': {
                       'b1': {
                           'c1': 'v1'
                           }
                       },
                'a2': {
                       'b2': {
                           'c2': 'v2'
                           }
                       },
                }
        },
        'key0': 'value0'
    }}
    dw = NestedMKDict(dct, recursive_to_others=True)
    dws = NestedMKDict(dct, sep='.', recursive_to_others=True)

    dwf = flatten(dw, ('a1', 'b1', 'c1', 'a2', 'b2', 'c2'))
    dwsf = flatten(dws, ('a1', 'b1', 'c1', 'a2', 'b2', 'c2'))

    for obj in (dwf, dwsf):
        assert obj['root', 'subfolder2', 'st', 'a1', 'b1', 'c1']=='v1'
        assert obj['root', 'subfolder2', 'st', 'b1', 'a1', 'c1']=='v1'
        assert obj['root', 'subfolder2', 'st', 'c1', 'a1', 'b1']=='v1'
        assert obj['root', 'subfolder2', 'st', 'a2', 'b2', 'c2']=='v2'
        assert obj['root', 'subfolder2', 'st', 'b2', 'a2', 'c2']=='v2'
        assert obj['root', 'subfolder2', 'st', 'c2', 'a2', 'b2']=='v2'

def test_nestedmkdict_flatten_v02():
    dct = {'root': {
        'subfolder1': {
            'key1': 'value1',
            'key2': 'value2'
        },
        'subfolder2': {
            'key1': 'value1',
            'key2': 'value2',
            'st': {
                'a1': {
                       'b1': {
                           'd1': 'extra',
                           'c1': 'v1'
                           }
                       },
                'a2': {
                       'b2': {
                           'c2': 'v2'
                           }
                       },
                }
        },
        'key0': 'value0'
    }}
    dw = NestedMKDict(dct, recursive_to_others=True)

    with raises(KeyError):
        dwf = flatten(dw, ('a1', 'b1', 'c1', 'a2', 'b2', 'c2'))
    # import IPython; IPython.embed(colors='neutral')
    # for obj in (dwf,):
    #     assert obj['root', 'subfolder2', 'st', 'a1', 'b1', 'c1']=='v1'
    #     assert obj['root', 'subfolder2', 'st', 'b1', 'a1', 'c1']=='v1'
    #     assert obj['root', 'subfolder2', 'st', 'c1', 'a1', 'b1']=='v1'
    #     assert obj['root', 'subfolder2', 'st', 'a2', 'b2', 'c2']=='v2'
    #     assert obj['root', 'subfolder2', 'st', 'b2', 'a2', 'c2']=='v2'
    #     assert obj['root', 'subfolder2', 'st', 'c2', 'a2', 'b2']=='v2'
    #     # FlatDict is unable to pass keys
    #     # assert obj['root', 'subfolder2', 'st', 'd1', 'a2', 'b2']=='extra'

def test_nestedmkdict_flatten_v03():
    dct = {'root': {
        'subfolder1': {
            'key1': 'value1',
            'key2': 'value2'
        },
        'subfolder2': {
            'key1': 'value1',
            'key2': 'value2',
            'st': {
                'a1': {
                       'b1': {
                           'c1': 'v1'
                           }
                       },
                'a2': {
                       'b2': {
                           'c2': 'v2',
                           'd1': 'extra'
                           }
                       },
                }
        },
        'key0': 'value0'
    }}
    dw = NestedMKDict(dct, recursive_to_others=True)
    dwf = flatten(dw, ('a1', 'b1', 'c1', 'a2', 'b2', 'c2'))
    # TODO: this test is insconsistent with test_nestedmkdict_flatten_v02
    # It does the same, but in different order.
    pprint(dwf.object)
    for obj in (dwf,):
        assert obj['root', 'subfolder2', 'st', 'a1', 'b1', 'c1']=='v1'
        assert obj['root', 'subfolder2', 'st', 'b1', 'a1', 'c1']=='v1'
        assert obj['root', 'subfolder2', 'st', 'c1', 'a1', 'b1']=='v1'
        assert obj['root', 'subfolder2', 'st', 'a2', 'b2', 'c2']=='v2'
        assert obj['root', 'subfolder2', 'st', 'b2', 'a2', 'c2']=='v2'
        assert obj['root', 'subfolder2', 'st', 'c2', 'a2', 'b2']=='v2'
        assert obj['root', 'subfolder2', 'st', 'd1', 'a2', 'b2']=='extra'

