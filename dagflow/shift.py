from itertools import zip_longest

from .exception import ConnectionError
from .iterators import iter_child_outputs, iter_inputs, iter_outputs

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
        fillvalue=None,
    ):
        if not output:
            raise ConnectionError("Unable to connect mismatching lists!")
        if isinstance(output, dict):
            if inp:
                raise ConnectionError(
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
        output.connect_to(inp)

    child_outputs = tuple(iter_child_outputs(inputs))
    return child_outputs[0] if len(child_outputs) == 1 else child_outputs


def lshift(inputs, outputs):
    """`<<` operator"""
    return rshift(outputs, inputs)
