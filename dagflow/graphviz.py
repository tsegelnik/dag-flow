from .input import Input
from .output import Output
from .printl import printl
from .types import NodeT

from numpy import square
from collections.abc import Sequence
from typing import Union, Set

try:
    import pygraphviz as G
except ImportError:
    GraphDot = None
    savegraph = None
else:

    def savegraph(graph, *args, **kwargs):
        gd = GraphDot(graph, **kwargs)
        gd.savegraph(*args)

    class GraphDot:
        _graph = None
        _node_id_map: dict

        _show: Set[str]
        def __init__(
            self,
            dag,
            graphattr: dict={}, edgeattr: dict={}, nodeattr: dict={},
            show: Union[Sequence,str] = ['type', 'mark', 'label'],
            **kwargs
        ):
            if show=='all' or 'all' in show:
                self._show = {'type', 'mark', 'label', 'status', 'data', 'data_summary'}
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
            self._nodes = {}
            self._nodes_open_input = {}
            self._nodes_open_output = {}
            self._edges = {}
            self._graph = G.AGraph(directed=True, strict=False, **kwargs)

            if graphattr:
                self._graph.graph_attr.update(graphattr)
            if edgeattr:
                self._graph.edge_attr.update(edgeattr)
            if nodeattr:
                self._graph.node_attr.update(nodeattr)

            if label := kwargs.pop("label", dag.label()):
                self.set_label(label)
            self._transform(dag)

        def _transform(self, dag):
            for nodedag in dag._nodes:
                self._add_node(nodedag)
            for nodedag in dag._nodes:
                self._add_open_inputs(nodedag)
                self._add_edges(nodedag)
            self.update_style()

        def get_id(self, object, suffix: str="") -> str:
            name = type(object).__name__
            omap = self._node_id_map.setdefault(name, {})
            onum = omap.setdefault(object, len(omap))
            return f"{name}_{onum}{suffix}"

        def get_label(self, node: NodeT) -> str:
            text = node.label() or node.name
            try:
                out0 = node.outputs[0]
            except IndexError:
                shape0 = '?'
                dtype0 = '?'
            else:
                shape0 = out0.shape
                if shape0 is None:
                    shape0 = '?'
                shape0="x".join(str(s) for s in shape0)

                dtype0 = out0.dtype
                if dtype0 is None:
                    dtype0 = '?'
                else:
                    dtype0 = dtype0.char

            npos = len(node.outputs)
            nall = node.outputs.len_all()
            if nall==npos:
                if npos>1:
                    nout = f' N{npos}'
                else:
                    nout = ''
            else:
                nout=f' N{npos}/{nall}'

            left, right = [], []
            info_type = f"[{shape0}]{dtype0}{nout}"
            if 'type' in self._show:
                left.append(info_type)
            if 'mark' in self._show and node.mark is not None:
                left.append(node.mark)
            if 'label' in self._show:
                right.append(text)
            if 'status' in self._show:
                status = []
                if node.types_tainted: status.append('types_tainted')
                if node.tainted: status.append('tainted')
                if node.frozen: status.append('frozen')
                if node.frozen_tainted: status.append('frozen_tainted')
                if node.invalid: status.append('invalid')
                if not node.closed: status.append('open')
                right.append(status)

            show_data = 'data' in self._show
            show_data_summary = 'data_summary' in self._show
            if show_data or show_data_summary:
                data = None
                tainted = out0.tainted and 'tainted' or 'updated'
                try:
                    data = out0.data
                except:
                    pass
                if data is None:
                    right.append('cought exception')
                else:
                    if show_data:
                        right.append(str(data).replace('\n', '\\l')+'\\l')
                    if show_data_summary:
                        sm = data.sum()
                        sm2 = square(data).sum()
                        mn = data.min()
                        mx = data.max()
                        right.append((f'Σ={sm:.2g}', f'Σ²={sm2:.2g}', f'min={mn:.2g}', f'max={mx:.2g}', f'{tainted}'))

            if node.exception is not None:
                right.append(node.exception)

            return self._combine_labels((left, right))

        def _combine_labels(self, labels: Union[Sequence,str]) -> str:
            if isinstance(labels, str):
                return labels

            slabels = [self._combine_labels(l) for l in labels]
            return f"{{{'|'.join(slabels)}}}"

        def _add_node(self, nodedag):
            styledict = {
                "shape": "Mrecord",
                "label": self.get_label(nodedag)
            }
            target = self.get_id(nodedag)
            self._graph.add_node(target, **styledict)
            nodedot = self._graph.get_node(target)
            self._nodes[nodedag] = nodedot

        def _add_open_inputs(self, nodedag):
            for input in nodedag.inputs:
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
            self._edges[input] = (nodein, edge, nodeout)

        def _add_edges(self, nodedag):
            for output in nodedag.outputs:
                if output.connected():
                    for input in output.child_inputs:
                        self._add_edge(nodedag, output, input)
                else:
                    self._add_open_output(nodedag, output)

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
            self._edges[output] = (nodein, edge, nodeout)

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

        def _add_edge(self, nodedag, output, input):
            styledict = {}

            source = self.get_id(nodedag)
            target = self.get_id(input.node)
            self._get_index(output, styledict, 'taillabel')
            self._get_index(input, styledict, 'headlabel')
            self._graph.add_edge(source, target, **styledict)

            nodein = self._graph.get_node(source)
            edge = self._graph.get_edge(source, target)
            nodeout = self._graph.get_node(target)

            self._edges[input] = (nodein, edge, nodeout)

        def _set_style_node(self, node, attr):
            if node is None:
                attr["color"] = "gray"
            else:
                if node.invalid:
                    attr["color"] = "black"
                elif node.being_evaluated:
                    attr["color"] = "gold"
                elif node.tainted:
                    attr["color"] = "red"
                elif node.frozen_tainted:
                    attr["color"] = "blue"
                elif node.frozen:
                    attr["color"] = "cyan"
                elif node.immediate:
                    attr["color"] = "green"
                else:
                    attr["color"] = "forestgreen"

                if node.exception is not None:
                    attr["color"] = "magenta"

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
                "arrowsize": 0.5,
                "arrowhead": allocated_on_input  and 'dotopen' or 'odotopen',
                "arrowtail": allocated_on_output and 'dot' or 'odot'
                })

            if node:
                if node.frozen:
                    attrin["style"] = "dashed"
                    attr["style"] = "dashed"
                    # attr['arrowhead']='tee'
                else:
                    attr["style"] = ""

        def update_style(self):
            for nodedag, nodedot in self._nodes.items():
                self._set_style_node(nodedag, nodedot.attr)

            for object, (nodein, edge, nodeout) in self._edges.items():
                self._set_style_edge(
                    object, nodein.attr, edge.attr, nodeout.attr
                )

        def set_label(self, label):
            self._graph.graph_attr["label"] = label

        def savegraph(self, fname, verbose=True):
            if verbose:
                printl("Write output file:", fname)

            if fname.endswith(".dot"):
                self._graph.write(fname)
            else:
                self._graph.layout(prog="dot")
                self._graph.draw(fname)
