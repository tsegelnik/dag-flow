
from gindex import GNIndex

def test_init():
	gi1 = GNIndex.from_dict({
		'a': ('a1', 'a2', 'a3'),
		'b': ('b1', 'b2', 'b3'),
		'c': ('c1', 'c2', 'b3'),
		})

	gi2 = GNIndex.from_dict({
		('a', 'alpha'): ('a1', 'a2', 'a3'),
		('b', 'beta'): ('b1', 'b2', 'b3'),
		'c': ('c1', 'c2', 'b3'),
		})


	check = [
			('a1', 'b1', 'c1'),
			('a1', 'b1', 'c2'),
			]
	for idx, cmpto in zip(gi2, check):
		assert idx.values==cmpto
