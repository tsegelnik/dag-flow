from __future__ import print_function

import itertools as I

from .tools import undefinedleg
from .iterators import iter_corresponding_outputs, iter_inputs, iter_outputs

# Python2 compatibility
zip_longest = getattr(I, "zip_longest", None)
if not zip_longest:
    zip_longest = getattr(I, "izip_longest")

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

    corresponding_outputs = tuple(iter_corresponding_outputs(inputs))

    if len(corresponding_outputs) == 1:
        return corresponding_outputs[0]

    return corresponding_outputs


def lshift(inputs, outputs):
    return rshift(outputs, inputs)
