from .input import Input
from .output import Output
from .types import NodeT
from .logger import logger, SUBINFO

from numpy import square
from typing import Union, Set, Optional, Dict, Sequence, Literal

from .graph import Graph
from .node import Node

try:
    import pygraphviz as G
except ImportError:
    GraphDot = None
    savegraph = None
else:

    def savegraph(graph, *args, **kwargs):
        gd = GraphDot(graph, **kwargs)
        gd.savegraph(*args)

    class EdgeDef:
        __slots__ = ('nodein', 'nodemid', 'nodeout', 'edges')
        def __init__(self, nodeout, nodemid, nodein, edge):
            self.nodein = nodein
            self.nodemid = nodemid
            self.nodeout = nodeout
            self.edges = [edge]

        def append(self, edge):
            self.edges.append(edge)

    class GraphDot:
        __slots__ = (
                '_graph',
                '_node_id_map',
                '_show',
                '_nodes_map_dag',
                '_nodes_open_input',
                '_nodes_open_output',
                '_edges'
                )
        _graph: G.AGraph
        _node_id_map: dict
        _nodes_map_dag: Dict[Node, G.agraph.Node]

        _show: Set[Literal['type', 'mark', 'label', 'path', 'status', 'data', 'data_summary']]
        def __init__(
            self,
            graph_or_node: Union[Graph, Node, None],
            graphattr: dict={}, edgeattr: dict={}, nodeattr: dict={},
            show: Union[Sequence,str] = ['type', 'mark', 'label'],
            **kwargs
        ):
            if show=='all' or 'all' in show:
                self._show = {'type', 'mark', 'label', 'path', 'status', 'data', 'data_summary'}
            else:
                self._show = set(show)

            graphattr = dict(graphattr)
            graphattr.setdefault("rankdir", "LR")
            graphattr.setdefault("dpi", 300)

            edgeattr = dict(edgeattr)
            edgeattr.setdefault("fontsize", 10)
            edgeattr.setdefault("labelfontsize", 9)
            edgeattr.setdefault("labeldistance", 1.2)

            nodeattr = dict(nodeattr)

            self._node_id_map = {}
            self._nodes_map_dag = {}
            self._nodes_open_input = {}
            self._nodes_open_output = {}
            self._edges: Dict[str, EdgeDef] = {}
            self._graph = G.AGraph(directed=True, strict=False, **kwargs)

            if graphattr:
                self._graph.graph_attr.update(graphattr)
            if edgeattr:
                self._graph.edge_attr.update(edgeattr)
            if nodeattr:
                self._graph.node_attr.update(nodeattr)

            if isinstance(graph_or_node, Graph):
                if label := kwargs.pop("label", graph_or_node.label()):
                    self.set_label(label)
                self._transform_graph(graph_or_node)
            elif isinstance(graph_or_node, Node):
                self._transform_from_node(graph_or_node)
            elif graph_or_node!=None:
                raise RuntimeError("Invalid graph entry point")

        @classmethod
        def from_graph(cls, graph: Graph, *args, **kwargs) -> 'GraphDot':
            gd = cls(None, *args, **kwargs)
            if (label := kwargs.pop("label", graph.label())):
                gd.set_label(label)
            gd._transform_graph(graph)
            return gd

        def _transform_graph(self, dag: Graph) -> None:
            for nodedag in dag._nodes:
                # if nodedag.meta_node:
                #     self._add_node(nodedag.meta_node)
                # else:
                self._add_node(nodedag)
            for nodedag in dag._nodes:
                self._add_open_inputs(nodedag)
                self._add_edges(nodedag)
            self.update_style()

        @classmethod
        def from_output(cls, output: Output, *args, **kwargs) -> 'GraphDot':
            return cls.from_node(output.node, *args, **kwargs)

        @classmethod
        def from_node(
            cls,
            node: Node,
            *args,
            mindepth: Optional[int] = None,
            maxdepth: Optional[int] = None,
            minsize: Optional[int] = None,
            **kwargs
        ) -> 'GraphDot':
            gd = cls(None, *args, **kwargs)
            label = [node.name]
            if mindepth is not None: label.append(f'{mindepth=:+d}')
            if maxdepth is not None: label.append(f'{maxdepth=:+d}')
            if minsize is not None: label.append(f'{minsize=:d}')
            gd.set_label(', '.join(label))
            gd._transform_from_node(node, mindepth=mindepth, maxdepth=maxdepth, minsize=minsize)
            return gd

        def _transform_from_node(
            self,
            node: Node,
            mindepth: Optional[int] = None,
            maxdepth: Optional[int] = None,
            minsize: Optional[int] = None,
            no_forward: bool = False,
            no_backward: bool = False,
        ) -> None:
            self._add_nodes_backward_recursive(
                node,
                including_self=True,
                mindepth=mindepth,
                maxdepth=maxdepth,
                minsize=minsize,
                no_forward=no_forward,
                no_backward=no_backward,
            )

            for nodedag in self._nodes_map_dag:
                self._add_open_inputs(nodedag)
                self._add_edges(nodedag)

            self.update_style()

        def _add_node(self, nodedag: Node, *, depth: Optional[int]=None) -> None:
            if nodedag in self._nodes_map_dag:
                return

            styledict = {
                "shape": "Mrecord",
                "label": self.get_label(nodedag, depth=depth)
            }
            target = self.get_id(nodedag)
            self._graph.add_node(target, **styledict)
            nodedot = self._graph.get_node(target)
            nodedot.attr['nodedag'] = nodedag
            nodedot.attr['depth'] = depth
            self._nodes_map_dag[nodedag] = nodedot

        def _add_nodes_backward_recursive(
            self,
            node: Node,
            *,
            including_self: bool=False,
            mindepth: Optional[int] = None,
            maxdepth: Optional[int] = None,
            minsize: Optional[int] = None,
            no_forward: bool = False,
            no_backward: bool = False,
            depth: int=0,
            visited_nodes: Set[Node] = set()
        ) -> None:
            if no_forward and no_backward:
                raise RuntimeError('May not set no_forward and no_backward simultaneously')
            if node not in visited_nodes:
                visited_nodes.add(node)

            if including_self:
                if node in self._nodes_map_dag:
                    return
                if not num_in_range(depth, mindepth, maxdepth):
                    return
                if depth>0 or num_in_range(node.outputs[0].dd.size, minsize):
                    self._add_node(node, depth=depth)
                else:
                    return
            depth-=1
            if not no_backward:
                for input in node.inputs.iter_all():
                    self._add_nodes_backward_recursive(
                        input.parent_node,
                        including_self=True,
                        depth=depth,
                        mindepth=mindepth,
                        maxdepth=maxdepth,
                        minsize=minsize,
                        no_forward=no_forward,
                        visited_nodes=visited_nodes
                    )

            if not no_forward:
                self._add_nodes_forward_recursive(
                    node,
                    including_self=False,
                    depth=depth+1,
                    no_backward=no_backward,
                    mindepth=mindepth,
                    maxdepth=maxdepth,
                    minsize=minsize,
                    ignore_visit=True,
                    visited_nodes=visited_nodes
                )

        def _add_nodes_forward_recursive(
            self,
            node: Node,
            *,
            including_self: bool=False,
            mindepth: Optional[int] = None,
            maxdepth: Optional[int] = None,
            minsize: Optional[int] = None,
            no_backward: bool = False,
            depth: int=0,
            visited_nodes: Set[Node] = set(),
            ignore_visit: bool = False
        ) -> None:
            if node in visited_nodes and not ignore_visit:
                return
            visited_nodes.add(node)

            if including_self:
                if node in self._nodes_map_dag:
                    return
                if not num_in_range(depth, mindepth, maxdepth):
                    return
                if depth>0 or num_in_range(node.outputs[0].dd.size, minsize):
                    self._add_node(node, depth=depth)
                else:
                    return
            depth+=1
            for output in node.outputs.iter_all():
                for child_input in output.child_inputs:
                    if not no_backward:
                        self._add_nodes_backward_recursive(
                            child_input.node,
                            including_self=True,
                            depth=depth,
                            mindepth=mindepth,
                            maxdepth=maxdepth,
                            minsize=minsize,
                            visited_nodes=visited_nodes
                        )
                    self._add_nodes_forward_recursive(
                        child_input.node,
                        including_self=no_backward,
                        depth=depth,
                        no_backward=no_backward,
                        mindepth=mindepth,
                        maxdepth=maxdepth,
                        minsize=minsize,
                        visited_nodes=visited_nodes,
                        ignore_visit=True
                    )

        def _add_open_inputs(self, nodedag):
            for input in nodedag.inputs.iter_all():
                if not input.connected():
                    self._add_open_input(input, nodedag)

        def _add_open_input(self, input, nodedag):
            styledict = {}
            source = self.get_id(input, "_in")
            target = self.get_id(nodedag)

            self._graph.add_node(source, label="", shape="none", **styledict)
            self._graph.add_edge(source, target, **styledict)

            nodein = self._graph.get_node(source)
            edge = self._graph.get_edge(source, target)
            nodeout = self._graph.get_node(target)

            self._nodes_open_input[input] = nodein
            self._edges[input] = EdgeDef(nodein, None, nodeout, edge)

        def _add_edges(self, nodedag):
            for output in nodedag.outputs.iter_all():
                if output.connected():
                    if len(output.child_inputs)>1:
                        self._add_edges_multi(nodedag, output)
                    else:
                        self._add_edge(nodedag, output, output.child_inputs[0])
                else:
                    self._add_open_output(nodedag, output)

                if output.dd.axes_edges:
                    self._add_edge_hist(output)

        def _add_edges_multi(self, nodedag, output):
            vnode = self.get_id(output, "_mid")
            self._graph.add_node(vnode, label="", shape="none", width=0, height=0, penwidth=0, weight=10)
            firstinput = output.child_inputs[0]
            self._add_edge(nodedag, output, firstinput, vtarget=vnode)
            for input in output.child_inputs:
                self._add_edge(nodedag, output, input, vsource=vnode)

        def _add_edge_hist(self, output: Output) -> None:
            if output.dd.edges_inherited:
                return
            eoutput = output.dd.axes_edges[0]

            self._add_edge(eoutput.node, eoutput, output, style={'style': 'dotted'})

        def _add_open_output(self, nodedag, output):
            styledict = {}
            source = self.get_id(nodedag)
            target = self.get_id(output, "_out")
            self._get_index(output, styledict, 'taillabel')

            self._graph.add_node(target, label="", shape="none", **styledict)
            self._graph.add_edge(
                source, target, arrowhead="empty", **styledict
            )
            nodein = self._graph.get_node(source)
            edge = self._graph.get_edge(source, target)
            nodeout = self._graph.get_node(target)

            self._nodes_open_output[output] = nodeout
            self._edges[output] = EdgeDef(nodein, None, nodeout, edge)

        def _get_index(self, leg, styledict: dict, target: str):
            if isinstance(leg, Input):
                container = leg.node.inputs
            else:
                container = leg.node.outputs
            if container.len_all()<2:
                return

            try:
                idx = container.index(leg)
            except ValueError:
                pass
            else:
                styledict[target] = str(idx)

        def _add_edge(self, nodedag, output, input, *, vsource: Optional[str]=None, vtarget: Optional[str]=None, style: Optional[dict]=None) -> None:
            styledict = style or {}

            if vsource is not None:
                source = vsource
                styledict['arrowtail'] = 'none'
            else:
                source = self.get_id(nodedag)
                self._get_index(output, styledict, 'taillabel')

            if vtarget is not None:
                target = vtarget
                styledict['arrowhead'] = 'none'
            else:
                target = self.get_id(input.node)
                self._get_index(input, styledict, 'headlabel')

            self._graph.add_edge(source, target, **styledict)

            nodein = self._graph.get_node(source)
            edge = self._graph.get_edge(source, target)
            nodeout = self._graph.get_node(target)

            edgedef = self._edges.get(input, None)
            if edgedef is None:
                self._edges[input] = EdgeDef(nodein, None, nodeout, edge)
            else:
                edgedef.append(edge)

        def _set_style_node(self, node, attr):
            if node is None:
                attr["color"] = "gray"
            else:
                try:
                    if   node.invalid:         attr["color"] = "black"
                    elif node.being_evaluated: attr["color"] = "gold"
                    elif node.tainted:         attr["color"] = "red"
                    elif node.frozen_tainted:  attr["color"] = "blue"
                    elif node.frozen:          attr["color"] = "cyan"
                    elif node.immediate:       attr["color"] = "green"
                    else:                      attr["color"] = "forestgreen"

                    if node.exception is not None:
                        attr["color"] = "magenta"
                except AttributeError:
                    attr["color"] = "forestgreen"

        def _set_style_edge(self, obj, attrin, attr, attrout):
            if isinstance(obj, Input):
                if obj.connected():
                    node = obj.parent_output.node
                else:
                    node = None
                    self._set_style_node(node, attrin)
            else:
                node = obj.node
                self._set_style_node(node, attrout)

            self._set_style_node(node, attr)

            if isinstance(obj, Input):
                allocated_on_input = obj.owns_buffer
                try:
                    allocated_on_output = obj.parent_output.owns_buffer
                except AttributeError:
                    allocated_on_output = True
            elif isinstance(obj, Output):
                allocated_on_input = False
                allocated_on_output = obj.owns_buffer
            attr.update({
                "dir": "both",
                "arrowsize": 0.5
                })
            attr["arrowhead"] = attr["arrowhead"] or allocated_on_input  and 'dotopen' or 'odotopen'
            attr["arrowtail"] = attr["arrowtail"] or allocated_on_output and 'dot' or 'odot'

            if node:
                if node.frozen:
                    attrin["style"] = "dashed"
                    if attr["style"]!="dotted":
                        attr["style"] = "dashed"
                    # attr['arrowhead']='tee'
                elif attr["style"]=="dashed":
                    attr["style"] = ""

        def update_style(self):
            for nodedag, nodedot in self._nodes_map_dag.items():
                self._set_style_node(nodedag, nodedot.attr)

            for object, edgedef in self._edges.items():
                for edge in edgedef.edges:
                    self._set_style_edge(
                        object, edgedef.nodein.attr, edge.attr, edgedef.nodeout.attr
                    )

        def set_label(self, label: str):
            self._graph.graph_attr["label"] = label

        def savegraph(self, fname):
            logger.log(SUBINFO, f'Write: {fname}')
            if fname.endswith(".dot"):
                self._graph.write(fname)
            else:
                self._graph.layout(prog="dot")
                self._graph.draw(fname)

        def get_id(self, object, suffix: str="") -> str:
            name = type(object).__name__
            omap = self._node_id_map.setdefault(name, {})
            onum = omap.setdefault(object, len(omap))
            return f"{name}_{onum}{suffix}"

        def get_label(self, node: NodeT, *, depth: Optional[int]=None) -> str:
            text = node.labels.graph or node.name
            try:
                out0 = node.outputs[0]
            except IndexError:
                shape0 = '?'
                dtype0 = '?'
                hasedges = False
                hasnodes = False
                out0 = None
            else:
                hasedges = bool(out0.dd.axes_edges)
                hasnodes = bool(out0.dd.axes_nodes)
                shape0 = out0.dd.shape
                if shape0 is None:
                    shape0 = '?'
                shape0="x".join(str(s) for s in shape0)

                dtype0 = out0.dd.dtype
                if dtype0 is None:
                    dtype0 = '?'
                else:
                    dtype0 = dtype0.char

            nout_pos = len(node.outputs)
            nout_nonpos = node.outputs.len_all()-nout_pos
            if nout_nonpos==0:
                if nout_pos>1:
                    nout = f'→{nout_pos}'
                else:
                    nout = ''
            else:
                nout=f'→{nout_pos}+{nout_nonpos}'

            nin_pos = len(node.inputs)
            nin_nonpos = node.inputs.len_all() - nin_pos
            if nin_nonpos==0:
                if nin_pos>1:
                    nin = f'{nin_pos}→'
                else:
                    nin = ''
            else:
                nin=f'{nin_pos}+{nin_nonpos}→'

            nlimbs = f' {nin}{nout}'.replace('→→', '→')

            left, right = [], []
            if hasedges:
                br_left, br_right = '\\{', '\\}'
            else:
                br_left, br_right = '[', ']'
            if hasnodes:
                br_right+='…'
            info_type = f"{br_left}{shape0}{br_right}{dtype0}{nlimbs}"
            if 'type' in self._show:
                left.append(info_type)
            if 'mark' in self._show and (mark:=node.labels.mark) is not None:
                left.append(mark)
            if 'label' in self._show:
                right.append(text)
            if 'path' in self._show and (paths:=node.labels.paths):
                right.append(f'path: {paths[0]}')
            if 'status' in self._show:
                status = []
                try:
                    if node.types_tainted:  status.append('types_tainted')
                    if node.tainted:        status.append('tainted')
                    if node.frozen:         status.append('frozen')
                    if node.frozen_tainted: status.append('frozen_tainted')
                    if node.invalid:        status.append('invalid')
                    if not node.closed:     status.append('open')
                except AttributeError:
                    pass
                if status:
                    right.append(status)

            show_data = 'data' in self._show
            show_data_summary = 'data_summary' in self._show
            if (show_data or show_data_summary) and out0 is not None:
                data = None
                tainted = out0.tainted and 'tainted' or 'updated'
                try:
                    data = out0.data
                except Exception:
                    right.append('cought exception')
                    data = out0._data

                if show_data_summary:
                    sm = data.sum()
                    sm2 = square(data).sum()
                    mn = data.min()
                    mx = data.max()
                    block = [f'Σ={sm:.2g}', f'Σ²={sm2:.2g}', f'min={mn:.2g}', f'max={mx:.2g}', f'{tainted}']
                    if depth is not None:
                        block.append(f'd: {depth:+d}'.replace('-', '−'))
                    right.append(block)

                if show_data:
                    right.append(str(data).replace('\n', '\\l')+'\\l')

            if getattr(node, 'exception', None) is not None:
                right.append(node.exception)

            return self._combine_labels((left, right))

        def _combine_labels(self, labels: Union[Sequence,str]) -> str:
            if isinstance(labels, str):
                return labels

            slabels = [self._combine_labels(l) for l in labels]
            return f"{{{'|'.join(slabels)}}}"

def num_in_range(num: int, minnum: Optional[int], maxnum: Optional[int]=None) -> bool:
    if minnum is not None and num<minnum:
        return False
    if maxnum is not None and num>maxnum:
        return False
    return True
