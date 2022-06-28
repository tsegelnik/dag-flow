from __future__ import print_function

from itertools import zip_longest

from .tools import undefinedleg
from .iterators import iter_iinputs, iter_inputs, iter_outputs

_rshift_scope_id = 0


def rshift_scope_id():
    global _rshift_scope_id
    ret = _rshift_scope_id
    _rshift_scope_id += 1
    return ret


def rshift(outputs, inputs):
    scope_id = rshift_scope_id()

    for output, inp in zip_longest(
        iter_outputs(outputs),
        iter_inputs(inputs, True),
        fillvalue=undefinedleg,
    ):
        if not output:
            raise RuntimeError("Unable to connect mismatching lists")
        if not inp:
            missing_input_handler = getattr(
                inputs, "_missing_input_handler", lambda *args, **kwargs: None
            )
            if not (inp := missing_input_handler(scope=scope_id)):
                break
        output._connect_to(inp)
    iinputs = tuple(iter_iinputs(inputs))
    return iinputs[0] if len(iinputs) == 1 else iinputs


def lshift(inputs, outputs):
    return rshift(outputs, inputs)
