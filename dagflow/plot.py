from matplotlib.pyplot import (
    stairs,
    plot,
    colorbar as plot_colorbar,
    gca,
    gcf,
    sca,
    cm,
    close as closefig,
    savefig,
    show as showfig
)
from matplotlib.pyplot import Axes
from matplotlib import colormaps
from .output import Output
from .limbs import Limbs
from .types import EdgesLike, NodesLike
from .logger import logger, SUBINFO

from typing import Union, List, Optional, Tuple, Mapping
from numpy.typing import ArrayLike, NDArray
from numpy import asanyarray, meshgrid, zeros_like
from numpy.ma import array as masked_array

def _mask_if_needed(datain: ArrayLike, /, *, masked_value: Optional[float]=None) -> NDArray:
    data = asanyarray(datain)
    if masked_value is None:
        return data

    mask = (data==masked_value)
    return masked_array(data, mask=mask)

def _get_node_data(node: Limbs, *args, **kwargs) -> Tuple[Optional[Output], NDArray, EdgesLike, NodesLike]:
    return _get_output_data(node.outputs[0], *args, **kwargs)

def _get_output_data(output: Output, *args, **kwargs) -> Tuple[Output, NDArray, EdgesLike, NodesLike]:
    data = _mask_if_needed(output.data, *args, **kwargs)
    return output, data, output.dd.edges_arrays, output.dd.nodes_arrays

def _get_array_data(array: ArrayLike, *args, **kwargs) -> Tuple[Optional[Output], NDArray, EdgesLike, NodesLike]:
    data = _mask_if_needed(array, *args, **kwargs)
    return None, data, (), ()

def _get_data(object: Union[Output, Limbs, ArrayLike], *args, **kwargs) -> Tuple[Optional[Output], NDArray, EdgesLike, NodesLike]:
    if isinstance(object, Output):
        return _get_output_data(object, *args, **kwargs)
    elif isinstance(object, Limbs):
        return _get_node_data(object, *args, **kwargs)
    else:
        return _get_array_data(object, *args, **kwargs)

def plot_auto(
    object: Union[Output, Limbs, ArrayLike],
    *args,
    filter_kw: dict = {},
    show_path: bool = True,
    save: Optional[str] = None,
    close: bool = False,
    show: bool = False,
    save_kw: dict = {},
    **kwargs
) -> Tuple[tuple, ...]:
    output, array, edges, nodes = _get_data(object, **filter_kw)

    ndim = len(array.shape)
    if ndim==1:
        edges = edges[0] if edges else None
        nodes = nodes[0] if nodes else None
        ret = plot_array_1d(array, edges, nodes, *args, **kwargs)
    elif ndim==2:
        colorbar = kwargs.pop('colorbar', {})
        if colorbar==True:
            colorbar={}
        if isinstance(colorbar, Mapping):
            colorbar.setdefault('label', output.labels.axis)
        ret = plot_array_2d(array, edges, nodes, *args, colorbar=colorbar, **kwargs)
    else:
        raise RuntimeError(f"Do not know how to plot {ndim}d")

    if output is not None:
        annotate_axes(output, show_path=show_path)

    if save:
        logger.log(SUBINFO, f'Write: {save}')
        savefig(save, **save_kw)
    if show: showfig()
    if close: closefig()

    return ret

def annotate_axes(output: Output, /, ax: Optional[Axes]=None, *, show_path: bool=True) -> None:
    ax = ax or gca()
    labels = output.labels

    title = labels.plottitle
    xlabel = output.dd.axis_label(0) or labels.xaxis or 'Index'

    ylabel = labels.axis
    if output.dd.dim==2:
        zlabel = ylabel
        ylabel = output.dd.axis_label(1)
    else:
        zlabel = None

    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if zlabel:
        try:
            ax.set_zlabel(zlabel)
        except AttributeError:
            pass

    if show_path:
        path = labels.paths
        if not path:
            return

        fig = gcf()
        try:
            ax.text2D(0.05, 0.05, path[0], transform=fig.dpi_scale_trans)
        except AttributeError:
            ax.text(0.05, 0.05, path[0], transform=fig.dpi_scale_trans)

def plot_array_1d(
    array: NDArray,
    edges: Optional[NDArray],
    nodes: Optional[NDArray],
    *args, **kwargs
) -> Tuple[tuple, ...]:
    if edges is not None:
        return plot_array_1d_hist(array, edges, *args, **kwargs)
    elif nodes is not None:
        return plot_array_1d_vs(array, nodes, *args, **kwargs)
    else:
        return plot_array_1d_array(array, *args, **kwargs)

def plot_array_1d_hist(array: NDArray, edges: Optional[NDArray], *args, **kwargs) -> Tuple:
    return stairs(array, edges, *args, **kwargs)

def plot_array_1d_vs(array: NDArray, nodes: Optional[NDArray], *args, **kwargs) -> Tuple:
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

def plot_array_2d(
    array: NDArray,
    edges: EdgesLike,
    nodes: NodesLike,
    *args, **kwargs
) -> Tuple[tuple, ...]:
    if edges:
        plot_array_2d_hist(array, edges, *args, **kwargs)
    elif nodes:
        plot_array_2d_vs(array, nodes, *args, **kwargs)
    else:
        plot_array_2d_array(array, *args, **kwargs)

def plot_array_2d_hist(
    dZ: NDArray,
    edges: List[NDArray],
    *args,
    method: str = 'pcolormesh',
    **kwargs
) -> Tuple:
    fcn = {
            'pcolor': plot_array_2d_hist_pcolor,
            'pcolorfast': plot_array_2d_hist_pcolorfast,
            'pcolormesh': plot_array_2d_hist_pcolormesh,
            'imshow': plot_array_2d_hist_imshow,
            'matshow': plot_array_2d_hist_matshow,
            'bar3d': plot_array_2d_hist_bar3d,
            }.get(method, None)

    if fcn is None:
        raise RuntimeError(f'Invlid 2d hist method: {method}')

    return fcn(dZ, edges, *args, **kwargs)

def plot_array_2d_vs(
    array: NDArray,
    nodes: List[NDArray],
    *args,
    method: str = 'pcolormesh',
    **kwargs
) -> Tuple:
    fcn = {
            'surface': plot_array_2d_vs_surface,
            'wireframe': plot_array_2d_vs_wireframe,
            'pcolormesh': plot_array_2d_vs_pcolormesh,
            'pcolor': plot_array_2d_vs_pcolor
            }.get(method, None)
    if fcn is None:
        raise RuntimeError("unimplemented")

    return fcn(array, nodes, *args, **kwargs)

def plot_array_2d_array(
    array: NDArray,
    *args,
    **kwargs
) -> Tuple:
    kwargs.setdefault('aspect', 'auto')
    return plot_array_2d_hist_matshow(array, None, *args, **kwargs)

def plot_array_2d_hist_bar3d(
    dZ: NDArray,
    edges: List[NDArray],
    *args,
    cmap: Optional[str] = None,
    **kwargs
) -> Tuple:
    xedges, yedges = edges
    xw=xedges[1:]-xedges[:-1]
    yw=yedges[1:]-yedges[:-1]

    X, Y = meshgrid(xedges[:-1], yedges[:-1], indexing='ij')
    X, Y = X.ravel(), Y.ravel()

    dX, dY = meshgrid(xw, yw, indexing='ij')
    dX, dY, dZ = dX.ravel(), dY.ravel(), dZ.ravel()
    Z = zeros_like(dZ)

    _, cmapper = apply_colors(dZ, cmap, kwargs, 'color')
    colorbar = kwargs.pop('colorbar', False)

    ax = gca()
    res = ax.bar3d(X, Y, Z, dX, dY, dZ, *args, **kwargs)

    return _colorbar_or_not_3d(res, colorbar, dZ)

def plot_array_2d_hist_pcolorfast(Z: NDArray, edges: List[NDArray], *args, **kwargs) -> Tuple:
    xedges, yedges = edges
    return pcolorfast(xedges, yedges, Z.T, *args, **kwargs)

def plot_array_2d_hist_pcolormesh(Z: NDArray, edges: List[NDArray], *args, **kwargs) -> Tuple:
    x, y = meshgrid(edges[0], edges[1], indexing='ij')
    return pcolormesh(x, y, Z, *args, **kwargs)

def plot_array_2d_hist_pcolor(Z: NDArray, edges: List[NDArray], *args, **kwargs) -> Tuple:
    x, y = meshgrid(edges[0], edges[1], indexing='ij')
    return pcolor(x, y, Z, *args, **kwargs)

def plot_array_2d_hist_imshow(Z: NDArray, edges: EdgesLike=None, *args, **kwargs):
    if edges:
        xedges, yedges = edges
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        kwargs.setdefault('extent', extent)
    kwargs.setdefault('origin', 'lower')
    return imshow(Z.T, *args, **kwargs)

def plot_array_2d_hist_matshow(Z: NDArray, edges: Optional[EdgesLike]=None, *args, **kwargs):
    kwargs.setdefault('fignum', False)
    if edges:
        xedges, yedges = edges
        extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]
        kwargs.setdefault('extent', extent)
    return matshow(Z.T, *args, **kwargs)

def plot_array_2d_vs_pcolormesh(Z: NDArray, nodes: List[NDArray], *args, **kwargs) -> Tuple:
    x, y = nodes
    kwargs.setdefault('shading', 'nearest')
    return pcolormesh(x, y, Z, *args, **kwargs)

def plot_array_2d_vs_pcolor(Z: NDArray, nodes: List[NDArray], *args, **kwargs) -> Tuple:
    x, y = nodes
    kwargs.setdefault('shading', 'nearest')
    return pcolormesh(x, y, Z, *args, **kwargs)

def plot_array_2d_vs_surface(Z: NDArray, nodes: List[NDArray], *args, **kwargs) -> Tuple:
    return plot_surface(nodes[0], nodes[1], Z, *args, **kwargs)

def plot_array_2d_vs_wireframe(
    Z: NDArray,
    nodes: List[NDArray],
    *args,
    # facecolors: Optional[str] = None,
    cmap: Union[str, bool, None] = None,
    colorbar: Union[dict, bool] = False,
    **kwargs
) -> Tuple:
    X, Y = nodes

    ax = gca()
    if cmap is not None:
        colors, _ = apply_colors(Z, cmap, kwargs, 'facecolors')
        if colors is not None:
            kwargs['rcount']=Z.shape[0]
            kwargs['ccount']=Z.shape[1]
            kwargs['shade']=False
            res = ax.plot_surface(X, Y, Z, **kwargs)
            res.set_facecolor((0, 0, 0, 0))

            return _colorbar_or_not_3d(res, colorbar, Z)

    return ax.plot_wireframe(X, Y, Z, *args, **kwargs)

def _patch_with_colorbar(fcn, mode3d=False):
    '''Patch pyplot.function or ax.method by adding a "colorbar" option'''
    returner = mode3d and _colorbar_or_not_3d or _colorbar_or_not
    if isinstance(fcn, str):
        def newfcn(*args, cmap: Union[bool, str, None]=None, colorbar: Optional[bool]=None, **kwargs):
            ax = gca()
            actual_fcn = getattr(ax, fcn)
            kwargs['cmap'] = cmap==True and 'viridis' or cmap
            res = actual_fcn(*args, **kwargs)
            return returner(res, colorbar)
    else:
        def newfcn(*args, cmap: Union[bool, str, None]=None, colorbar: Optional[bool]=None, **kwargs):
            kwargs['cmap'] = cmap==True and 'viridis' or cmap
            res = fcn(*args, **kwargs)
            return returner(res, colorbar)

    return newfcn

def apply_colors(
    buf: NDArray,
    cmap: Union[str, bool, None],
    kwargs: dict,
    colorsname: str
) -> Tuple:

    if cmap==True:
        cmap='viridis'
    elif not cmap:
        return None, None

    bmin, bmax = buf.min(), buf.max()
    norm = (buf-bmin)/(bmax-bmin)

    cmap = colormaps.get_cmap(cmap)
    res = cmap(norm)
    kwargs[colorsname] = res
    return res, cmap

def add_colorbar(
    colormapable,
    rasterized: bool=True,
    minorticks: bool=False,
    minorticks_values: Optional[NDArray]=None,
    label: Optional[str]=None,
    **kwargs
):
    """Add a colorbar to the axis with height aligned to the axis"""
    ax = gca()
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = gcf().colorbar(colormapable, cax=cax, **kwargs)

    if minorticks:
        if isinstance(minorticks, str):
            if minorticks=='log':
                minorticks_values = colormapable.norm(minorticks_values)
            # elif minorticks=='linear':
            #     pass

            l1, l2 = cax.get_ylim()
            minorticks_values = minorticks_values[ (minorticks_values>=l1)*(minorticks_values<=l2) ]
            cax.yaxis.set_ticks(minorticks_values, minor=True)
        else:
            cax.minorticks_on()

    if rasterized:
        cbar.solids.set_rasterized( True )

    if label is not None:
        cbar.set_label(label, rotation=270, labelpad=15)
    sca( ax )
    return cbar

def add_colorbar_3d(res, cbaropt: dict={}, mappable=None):
    """Add a colorbar to the 3d axis with height aligned to the axis"""
    cbaropt.setdefault('aspect', 4)
    cbaropt.setdefault('shrink', 0.5)

    if mappable is None:
        cbar = plot_colorbar(res, **cbaropt)
    else:
        colourMap = cm.ScalarMappable()
        colourMap.set_array(mappable)
        cbar = plot_colorbar(colourMap, **cbaropt)

    return res, cbar

def _colorbar_or_not(res, cbaropt: Union[Mapping, bool, None]):
    if not cbaropt:
        return res

    if not isinstance(cbaropt, Mapping):
        cbaropt = {}

    cbar = add_colorbar(res, **cbaropt)

    return res, cbar

def _colorbar_or_not_3d(res, cbaropt: Union[Mapping, bool, None], mappable=None):
    if not cbaropt:
        return res

    if not isinstance(cbaropt, Mapping):
        cbaropt = {}

    cbaropt.setdefault('aspect', 4)
    cbaropt.setdefault('shrink', 0.5)

    if mappable is None:
        cbar = plot_colorbar(res, ax=gca(), **cbaropt)
    else:
        colourMap = cm.ScalarMappable()
        colourMap.set_array(mappable)
        cbar = plot_colorbar(colourMap, ax=gca(), **cbaropt)

    return res, cbar

from matplotlib.pyplot import pcolor, pcolormesh, imshow, matshow
pcolorfast = _patch_with_colorbar('pcolorfast')
pcolor     = _patch_with_colorbar(pcolor)
pcolormesh = _patch_with_colorbar(pcolormesh)
imshow     = _patch_with_colorbar(imshow)
matshow    = _patch_with_colorbar(matshow)
plot_surface = _patch_with_colorbar('plot_surface', mode3d=True)
