from typing import Optional, Union, Callable

def _make_formatter(fmt: Union[str, Callable, dict]) -> Callable:
    if isinstance(fmt, str):
        return fmt.format
    elif isinstance(fmt, dict):
        return lambda s: fmt.get(s, s)

    return fmt

def inherit_labels(
	source: dict,
	destination: Optional[dict]=None,
	*,
	fmtlong: Union[str, Callable],
	fmtshort: Union[str, Callable]
) -> dict:
    if destination is None:
        destination = {}

    fmtlong = _make_formatter(fmtlong)
    fmtshort = _make_formatter(fmtshort)

    kshort = {'mark'}
    kskip = {'key', 'name'}
    for k, v in source.items():
        if k in kskip:
            continue
        newv = fmtshort(v) if k in kshort else fmtlong(v)
        if newv is not None:
            destination[k] = newv

    return destination
