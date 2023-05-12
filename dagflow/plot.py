from matplotlib.pyplot import stairs, plot, gca
from matplotlib.pyplot import Axes
from .node import Node, Output

from typing import Union, List, Optional, Tuple
from numpy.typing import ArrayLike, NDArray
from numpy import asanyarray

def _get_node_data(node: Node) -> Tuple[Optional[Output], NDArray, Optional[List[NDArray]], Optional[List[NDArray]]]:
	return _get_output_data(node.outputs[0])

def _get_output_data(output: Output) -> Tuple[Optional[Output], NDArray, Optional[List[NDArray]], Optional[List[NDArray]]]:
	return output, output.data, output.dd.edges_arrays, output.dd.nodes_arrays

def _get_array_data(array: ArrayLike) -> Tuple[Optional[Output], NDArray, Optional[List[NDArray]], Optional[List[NDArray]]]:
	return None, asanyarray(array), None, None

def _get_data(object: Union[Output, Node, ArrayLike]) -> Tuple[Optional[Output], NDArray, Optional[List[NDArray]], Optional[List[NDArray]]]:
	if isinstance(object, Output):
		return _get_output_data(object)
	elif isinstance(object, Node):
		return _get_node_data(object)
	else:
		return _get_array_data(object)

def plot_auto(object: Union[Output, Node, ArrayLike], *args, **kwargs) -> Tuple[tuple, ...]:
	output, array, edges, nodes = _get_data(object)

	ndim = len(array.shape)
	if ndim==1:
		if edges is not None: edges = edges[0]
		if nodes is not None: nodes = nodes[0]
		ret = plot_array_1d(array, edges, nodes, *args, **kwargs)
	elif ndim==2:
		ret = plot_array_2d(array, edges, nodes, *args, **kwargs)
	else:
		raise RuntimeError(f"Do not know how to plot {ndim}d")

	if output is not None:
		annotate_axes(output)

	return ret

def annotate_axes(output: Output, ax: Optional[Axes]=None) -> None:
	ax = ax or gca()
	node = output.node

	title = node.label('title', fallback=('text'))
	xlabel = output.dd.axis_label(0)

	if output.dd.dim==2:
		ylabel = output.dd.axis_label(1)
	else:
		ylabel = node.label('axis', fallback=('title', 'text'))

	if title: ax.set_title(title)
	if xlabel: ax.set_xlabel(xlabel)
	if ylabel: ax.set_ylabel(ylabel)

def plot_array_1d(
	array: NDArray,
	edges: Optional[List[NDArray]],
	nodes: Optional[List[NDArray]],
	*args, **kwargs
) -> Tuple[tuple, ...]:
	wasplot = False
	rets = []
	if edges is not None:
		ret = plot_array_1d_hist(array, edges, *args, **kwargs)
		rets.append(ret)
		wasplot = True

	if nodes is not None:
		ret = plot_array_1d_vs(array, nodes, *args, **kwargs)
		rets.append(ret)
		wasplot = True

	if wasplot:
		return

	ret = plot_array_1d_array(array, *args, **kwargs)
	rets.append(ret)

	return tuple(rets)

def plot_array_1d_hist(array: NDArray, edges: Optional[List[NDArray]], *args, **kwargs) -> Tuple:
	return stairs(array, edges, *args, **kwargs)

def plot_array_1d_vs(array: NDArray, nodes: Optional[List[NDArray]], *args, **kwargs) -> Tuple:
	return plot(nodes, array, *args, **kwargs)

def plot_array_1d_array(array: NDArray, *args, **kwargs) -> Tuple:
	return plot(array, *args, **kwargs)

def plot_output_1d(output: Output, *args, **kwargs) -> Tuple:
	array, edges, nodes = _get_data(output)
	return plot_array_1d(array, edges, nodes, *args, **kwargs)

def plot_output_1d_vs(output: Output, args, **kwargs) -> Tuple:
	array, edges, _ = _get_data(output)
	return plot_array_1d_vs(array, edges, *args, **kwargs)

def plot_output_1d_nodes(output: Output, args, **kwargs) -> Tuple:
	array, _, nodes = _get_data(output)
	return plot_array_1d_vs(array, nodes, *args, **kwargs)

def plot_output_1d_array(output: Output, args, **kwargs) -> Tuple:
	return plot_array_1d_array(output.data, *args, **kwargs)
