from typing import Iterable
from itertools import islice

class StopNesting(Exception):
    def __init__(self, object):
        self.object = object

def IsIterable(obj):
    return isinstance(obj, Iterable) and not isinstance(obj, str)

