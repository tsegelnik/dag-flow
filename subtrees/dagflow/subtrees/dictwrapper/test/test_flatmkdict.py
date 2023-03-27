from itertools import permutations

from multikeydict.flatmkdict import FlatMKDict
from pytest import raises


def test_getset():
    flatmkdict = FlatMKDict()
    safeflatmkdict = FlatMKDict(protect=True)
    val = "val"
    val2 = ["val", "lav"]
    flatmkdict["a", "b", "c"] = val
    safeflatmkdict["a", "b", "c"] = val
    for key in permutations(("a", "b", "c")):
        assert flatmkdict[tuple(key)] == val
        assert safeflatmkdict[tuple(key)] == val
    flatmkdict["c", "b", "a"] = val2
    for key in permutations(("a", "b", "c")):
        assert flatmkdict[tuple(key)] == val2
        with raises(AttributeError):
            safeflatmkdict[tuple(key)] = val2
    safeflatmkdict._protect = False
    for key in permutations(("a", "b", "c")):
        safeflatmkdict[tuple(key)] = val


def test_slice_filter():
    flatmkdict = FlatMKDict()
    flatmkdict["a", "b"] = 1
    flatmkdict["a", "b", "c"] = 2
    flatmkdict["a", "c", "d", "b"] = 3
    assert all(
        len(tuple(x)) == 3
        for x in (flatmkdict.items(), flatmkdict.items("a"), flatmkdict.items("a", "b"))
    )
    assert len(tuple(flatmkdict.items("a", "b", "c"))) == 2
    assert len(tuple(flatmkdict.items("a", "b", "d", "c"))) == 1
    assert isinstance(flatmkdict.slice("a"), FlatMKDict)
    assert all(
        x == flatmkdict
        for x in (
            flatmkdict.slice("a"),
            flatmkdict.slice("a", "b"),
            flatmkdict.slice(
                filterkey=lambda key: all(elem in "abcd" for elem in key)
            ),
            flatmkdict.slice(filterkeyelem=lambda key: key in "abcd")
        )
    )
    assert flatmkdict.slice("a", "b", "c") == {
        ("a", "b", "c"): 2,
        ("a", "b", "c", "d"): 3,
    }
    assert flatmkdict.slice("a", "b", "c", "d") == {
        ("a", "b", "c", "d"): 3,
    }
    assert flatmkdict.slice(
        filterkey=lambda key: all(elem != "d" for elem in key)
    ) == {
        ("a", "b", "c"): 2,
        ("a", "b"): 1,
    }
