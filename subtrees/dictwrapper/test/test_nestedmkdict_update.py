from multikeydict.nestedmkdict import NestedMKDict

from pytest import raises

def test_nestedmkdict_update_01():
    dct1 = {
        'a': 1,
        'b': 2,
        'c': 3,
        'd': {
            'da': 4
        },
        'e': {
            'ea': {
                'eaa': 5
            }
        }
    }
    dct2 = {
        'c': 4,
        'd': {
            'da': 5,
            'db': 6
        },
        'i': {
            'ia': 7
            }
    }
    dct3 = {
        'd': {
            'da': {
                'daa': 1
                }
        }
    }
    dct4 = {
        'd': 6
    }
    dct1_u2 = {
            'a': 1,
            'b': 2,
            'c': 4,
            'd': {
                'da': 5,
                'db': 6
                },
            'e': {
                'ea': {
                    'eaa': 5
                    }
                },
            'i': {
                'ia': 7
                }
            }
    dw1a = NestedMKDict(dct1)
    dw2 = NestedMKDict(dct2)
    dw3 = NestedMKDict(dct3)
    dw4 = NestedMKDict(dct3)

    dw1 = dw1a.deepcopy()
    dw1.update(dw2)
    assert dw1==dct1_u2

    dw1 = dw1a.deepcopy()
    dw1|=dw2
    assert dw1==dct1_u2

    dw1 = dw1a.deepcopy()
    with raises(TypeError):
        dw1|=dw3

    dw1 = dw1a.deepcopy()
    with raises(TypeError):
        dw1^=dw3

    dw1 = dw1a.deepcopy()
    with raises(TypeError):
        dw1^=dw2

    dw1 = dw1a.deepcopy()
    with raises(TypeError):
        dw1|=dw4

    dw1 = dw1a.deepcopy()
    with raises(TypeError):
        dw1^=dw4
