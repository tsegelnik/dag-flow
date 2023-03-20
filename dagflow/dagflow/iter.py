from collections.abc import Iterable
from itertools import islice

class StopNesting(Exception):
    def __init__(self, object):
        self.object = object

def IsIterable(obj):
    return isinstance(obj, Iterable) and not isinstance(obj, str)

def nth(iterable, n):
    "Returns the nth item or a default value"
    return next(islice(iterable, n, None)) if n > -1 else tuple(iterable)[n]

