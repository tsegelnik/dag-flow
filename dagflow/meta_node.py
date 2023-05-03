from .node import Node
from .limbs import Limbs

from typing import Sequence, List, Optional

class MetaNode(Limbs):
	"""A node containing multiple nodes and exposing part of their inputs and outputs"""
	__slots__ = ('_nodes',)

	_nodes: List[Node]
	_node_inputs_pos: Optional[Node]
	_node_outputs_pos: Optional[Node]

	def __init__(self):
		# super().__init__()

		self._nodes = []
		self._node_inputs_pos = None
		self._node_outputs_pos = None

	def add_node(
		self,
		node: Node,
		*,
		inputs_pos: bool=False,
		outputs_pos: bool=False,
		inputs_kw: Sequence[str]=[],
		outputs_kw: Sequence[str]=[]
	) -> None:
		if node in self._nodes:
			raise RuntimeError("Node already added")

		self._nodes.append(node)
		node.meta_node = self

		if inputs_pos:
			if self._node_inputs_pos is not None:
				raise RuntimeError("inputs_Pos already inherited")
			self._node_inputs_pos = node
			for input in node.inputs:
				self.inputs.add(input, True)

		for iname in inputs_kw:
			self.inputs.add(node.inputs.get_kw(iname))

		if outputs_pos:
			if self._node_outputs_pos is not None:
				raise RuntimeError("outputs_Pos already inherited")
			self._node_outputs_pos = node
			for output in node.outputs:
				self.outputs.add(output, True)

		for oname in outputs_kw:
			self.outputs.add(node.outputs.get_kw(oname))
