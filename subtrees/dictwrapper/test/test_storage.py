from itertools import permutations

from storage.storage import Storage
from pytest import raises


def test_getset():
    storage = Storage()
    safestorage = Storage(protect=True)
    val = "val"
    val2 = ["val", "lav"]
    storage["a", "b", "c"] = val
    safestorage["a", "b", "c"] = val
    for key in permutations(("a", "b", "c")):
        assert storage[tuple(key)] == val
        assert safestorage[tuple(key)] == val
    storage["c", "b", "a"] = val2
    for key in permutations(("a", "b", "c")):
        assert storage[tuple(key)] == val2
        with raises(AttributeError):
            safestorage[tuple(key)] = val2
    safestorage._protect = False
    for key in permutations(("a", "b", "c")):
        safestorage[tuple(key)] = val


def test_slice_filter():
    storage = Storage()
    storage["a", "b"] = 1
    storage["a", "b", "c"] = 2
    storage["a", "c", "d", "b"] = 3
    assert all(
        len(tuple(x)) == 3
        for x in (storage.items(), storage.items("a"), storage.items("a", "b"))
    )
    assert len(tuple(storage.items("a", "b", "c"))) == 2
    assert len(tuple(storage.items("a", "b", "d", "c"))) == 1
    assert isinstance(storage.slice("a"), Storage)
    assert all(
        x == storage
        for x in (
            storage.slice("a"),
            storage.slice("a", "b"),
            storage.slice(
                filterkey=lambda key: all(elem in "abcd" for elem in key)
            ),
            storage.slice(filterkeyelem=lambda key: key in "abcd")
        )
    )
    assert storage.slice("a", "b", "c") == {
        ("a", "b", "c"): 2,
        ("a", "b", "c", "d"): 3,
    }
    assert storage.slice("a", "b", "c", "d") == {
        ("a", "b", "c", "d"): 3,
    }
    assert storage.slice(
        filterkey=lambda key: all(elem != "d" for elem in key)
    ) == {
        ("a", "b", "c"): 2,
        ("a", "b"): 1,
    }
