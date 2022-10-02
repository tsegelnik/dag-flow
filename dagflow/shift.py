from __future__ import print_function

from itertools import zip_longest

from .tools import undefined
from .iterators import iter_parent_outputs, iter_inputs, iter_outputs

_rshift_scope_id = 0


def rshift_scope_id():
    global _rshift_scope_id
    ret = _rshift_scope_id
    _rshift_scope_id += 1
    return ret


def rshift(outputs, inputs):
    """`>>` operator"""
    scope_id = rshift_scope_id()

    for output, inp in zip_longest(
        iter_outputs(outputs),
        iter_inputs(inputs, True),
        fillvalue=undefined("leg"),
    ):
        if not output:
            raise RuntimeError("Unable to connect mismatching lists!")
        # NOTE: Now works only if the `inputs` is a single node
        # In other cases it is ambiguous
        if isinstance(output, dict):
            if inp:
                raise RuntimeError(
                    f"Cannot perform a binding from dict={output} due to "
                    f"non-empty input={inp}!"
                )
            for key, val in output.items():
                val >> inputs(key)
            continue
        if not inp:
            missing_input_handler = getattr(
                inputs, "_missing_input_handler", lambda *args, **kwargs: None
            )
            if not (inp := missing_input_handler(scope=scope_id)):
                break
        output._connect_to(inp)
    parent_outputs = tuple(iter_parent_outputs(inputs))
    return parent_outputs[0] if len(parent_outputs) == 1 else parent_outputs


def lshift(inputs, outputs):
    """`<<` operator"""
    return rshift(outputs, inputs)
