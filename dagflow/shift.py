from itertools import zip_longest

from .exception import ConnectionError
from .iterators import iter_child_outputs, iter_inputs, iter_outputs

from itertools import repeat

_rshift_scope_id = 0


def rshift_scope_id():
    global _rshift_scope_id
    ret = _rshift_scope_id
    _rshift_scope_id += 1
    return ret


def rshift(outputs, inputs):
    """`>>` operator"""
    scope_id = rshift_scope_id()

    outputs_all = tuple(iter_outputs(outputs))
    inputs_all = tuple(iter_inputs(inputs, disconnected_only=True))

    if len(outputs_all)==1 and len(inputs_all)!=len(outputs_all):
        permit_multiple_expansion = False
        outputs_it = repeat(outputs_all[0])
    else:
        permit_multiple_expansion = True
        outputs_it = iter(outputs_all)

    already_expanded = False
    for output, inp in zip_longest(
        outputs_it,
        inputs_all,
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
        if inp is None:
            if not permit_multiple_expansion and already_expanded:
                break

            missing_input_handler = getattr(
                inputs, "_missing_input_handler", lambda *args, **kwargs: None
            )
            already_expanded = True
            if not (inp := missing_input_handler(scope=scope_id)):
                break
        output.connect_to(inp)

    child_outputs = tuple(iter_child_outputs(inputs))
    return child_outputs[0] if len(child_outputs) == 1 else child_outputs

# def lshift(inputs, outputs):
#     """`<<` operator"""
#     return rshift(outputs, inputs)
