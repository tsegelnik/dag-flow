from .nestedmkdict import NestedMKDict
from .flatmkdict import FlatMKDict

from typing import Sequence, Tuple

def _select(seq: Sequence, elems_mask: set) -> Tuple[Tuple, Tuple]:
	selected = []
	rest = ()
	for i, el in enumerate(reversed(seq), 0):
		if el in elems_mask:
			selected.append(el)
		else:
			rest = tuple(seq[:len(seq)-i])
			break
	return tuple(rest), tuple(reversed(selected))

def flatten(mkdict, selkeys: Sequence=()) -> NestedMKDict:
	selkeys_set = set(selkeys)
	newdict = mkdict._wrap({}, parent=mkdict)
	newdict._parent = None

	for key, v in mkdict.walkitems():
		keys_nested, keys_flat = _select(key, selkeys_set)
		if keys_flat:
			flatdict = newdict.get(keys_nested, None)
			if flatdict is None:
				newdict[keys_nested] = (flatdict:=FlatMKDict(((keys_flat, v),),))
			elif isinstance(flatdict, FlatMKDict):
				flatdict[keys_flat] = v
			else:
				raise KeyError(f'Unable to flatten: {".".join(key)}')
		else:
			newdict[key] = v

	return newdict
