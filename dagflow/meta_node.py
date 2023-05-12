from .node import Node
from .limbs import Limbs

from typing import Sequence, List, Optional, Union, Tuple, Callable, Dict

class MetaNode(Limbs):
    """A node containing multiple nodes and exposing part of their inputs and outputs"""
    __slots__ = ('_nodes', '_node_inputs_pos', '_node_inputs_pos', '_missing_input_handler')

    _nodes: List[Node]
    _node_inputs_pos: Optional[Node]
    _node_outputs_pos: Optional[Node]
    _missing_input_handler: Callable

    def __init__(self):
        super().__init__()

        self._nodes = []
        self._node_inputs_pos = None
        self._node_outputs_pos = None
        self._missing_input_handler = lambda *_, **__: None

    def add_node(
        self,
        node: Node,
        *,
        inputs_pos: bool=False,
        outputs_pos: bool=False,
        kw_inputs: Sequence[Union[str, Tuple[str, str], Dict[str, str]]]=[],
        kw_outputs: Sequence[Union[str, Tuple[str, str], Dict[str, str]]]=[],
        merge_inputs: Sequence[str]=[],
        missing_inputs: bool=False,
        also_missing_outputs: bool=False,
    ) -> None:
        if node in self._nodes:
            raise RuntimeError("Node already added")

        self._nodes.append(node)
        node.meta_node = self

        if inputs_pos: self.import_pos_inputs(node)
        if outputs_pos: self.import_pos_outputs(node)
        self.import_kw_inputs(node, kw_inputs, merge=merge_inputs)
        self.import_kw_outputs(node, kw_outputs)

        if missing_inputs:
            self._missing_input_handler = MissingInputInherit(node, self, inherit_outputs=also_missing_outputs)
        if not missing_inputs and also_missing_outputs:
            raise RuntimeError('also_missiong_outputs=True option makes no sense')

    def import_pos_inputs(self, node: Node) -> None:
        if self._node_inputs_pos is not None:
            raise RuntimeError("Positional inputs already inherited")
        self._node_inputs_pos = node
        for input in node.inputs:
            self.inputs.add(input, positional=True, keyword=True)

    def import_pos_outputs(self, node: Node) -> None:
        if self._node_outputs_pos is not None:
            raise RuntimeError("Positional outputs already inherited")
        self._node_outputs_pos = node
        for output in node.outputs:
            self.outputs.add(output, positional=True, keyword=True)

    def import_kw_inputs(
        self,
        node: Node,
        kw_inputs: Sequence[Union[str, Tuple[str, str], Dict[str, str]]]=[],
        merge: Sequence[str]=[]
    ) -> None:
        if isinstance(kw_inputs, dict):
            iterable = kw_inputs.items()
        else:
            iterable = kw_inputs
        for iname in iterable:
            tname = iname
            if not isinstance(iname, str):
                iname, tname = iname
            newinput = node.inputs.get_kw(iname)
            mergethis = tname in merge
            self.inputs.add(newinput, name=tname, merge=mergethis, positional=False)

    def import_kw_outputs(self, node: Node, kw_outputs: Sequence[Union[str, Tuple[str, str], Dict[str, str]]]=[]) -> None:
        if isinstance(kw_outputs, dict):
            iterable = kw_outputs.items()
        else:
            iterable = kw_outputs

        for oname in iterable:
            tname = None
            if not isinstance(oname, str):
                oname, tname = oname
            self.outputs.add(node.outputs.get_kw(oname), name=tname, positional=False)

    def print(self, recursive: bool=False):
        print(f"MetaNode: →[{len(self.inputs)}],[{len(self.outputs)}]→")
        for i, input in enumerate(self.inputs):
            print("  ", i, input)
        for name, input in self.inputs.items_nonpos():
            print(f"     {input} [{name}]")
        for i, output in enumerate(self.outputs):
            print("  ", i, output)
        for name, input in self.outputs.items_nonpos():
            print(f"     {output} [{name}]")

        if recursive:
            for i, node in enumerate(self._nodes):
                print(f'subnode {i}: ', end='')
                node.print()

class MissingInputInherit:
    __slots__ = ('_source_node', '_target_node', '_source_handler', '_inherit_outputs')
    _source_node: Node
    _target_node: MetaNode
    _source_handler: Callable
    _inherit_outputs: bool

    def __init__(
        self,
        source_node: Node,
        target_node: Node,
        *,
        inherit_outputs: bool=False
    ):
        self._source_node = source_node
        self._target_node = target_node
        self._inherit_outputs = inherit_outputs

        try:
            self._source_handler = source_node._missing_input_handler
        except AttributeError:
            raise RuntimeError(f"Node {source_node!s} has no missing input handler")

    def __call__(self, *args, **kwargs):
        newinput = self._source_handler(*args, **kwargs)
        self._target_node.inputs.add(newinput)

        if self._inherit_outputs and newinput.child_output is not None:
            self._target_node.outputs.add(newinput.child_output)

        return newinput
