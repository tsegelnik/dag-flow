from __future__ import print_function
from .tools import IsIterable, StopNesting

def get_proper_iterator(obj, methodname, onerror, **kwargs):
    if methodname:
        if method := getattr(obj, methodname, None):
            return method(**kwargs)

    if isinstance(obj, dict):
        return obj.values()
    elif IsIterable(obj):
        return obj

    raise RuntimeError(f'Do not know how to get iterator for {onerror}')

def deep_iterate(obj, methodname, onerror, **kwargs):
    try:
        iterable = get_proper_iterator(obj, methodname, onerror, **kwargs)
        for element in iterable:
            yield from deep_iterate(element, methodname, onerror, **kwargs)
    except StopNesting as sn:
        yield sn.object

def iterate_values_or_list(obj):
    if isinstance(obj, dict):
        return obj.values()
    elif IsIterable(obj):
        return obj

def iter_inputs(inputs, disconnected_only=False):
    return deep_iterate(inputs, '_deep_iter_inputs', 'inputs', disconnected_only=disconnected_only)

def iter_outputs(outputs):
    return deep_iterate(outputs, '_deep_iter_outputs', 'outputs')

def iter_corresponding_outputs(inputs):
    return deep_iterate(inputs, '_deep_iter_corresponding_outputs', 'corresponding outputs')

