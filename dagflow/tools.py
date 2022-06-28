from __future__ import print_function

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


class Undefined:
    def __init__(self, what):
        self.what = what

    def __str__(self):
        return f"Undefined {self.what}"

    def __repr__(self):
        return f'Undefined("{self.what}")'

    def __bool__(self):
        return False

    def __nonzero__(self):
        """Python2 compatibility"""
        return False

    def __call__(self, *args, **kwargs):
        pass


undefinedname = Undefined("name")
undefineddata = Undefined("data")
undefineddatatype = Undefined("datatype")
undefinednode = Undefined("node")
undefinedgraph = Undefined("graph")
undefinedoutput = Undefined("output")
undefinedinput = Undefined("intput")
undefinedleg = Undefined("leg")
undefinedfunction = Undefined("function")
