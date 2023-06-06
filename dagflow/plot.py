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
from .node import Node
from .limbs import Limbs
from .types import EdgesLike, MeshesLike
from .logger import logger, SUBINFO

from typing import Union, List, Optional, Tuple, Mapping, Literal
from numpy.typing import ArrayLike, NDArray
from numpy import asanyarray, meshgrid, zeros_like
from numpy.ma import array as masked_array

class plot_auto:
    __slots__ = (
        '_object',
        '_output',
        '_array',
        '_edges',
        '_meshes',
        '_plotmethod',
        '_ret',
        '_title',
        '_xlabel',
        '_ylabel',
        '_zlabel'
    )
    _object: Union[Node, Output, NDArray]
    _output: Optional[Output]
    _array: NDArray
    _edges: EdgesLike
    _meshes: MeshesLike
    _plotmethod: Optional[str]
    _ret: Tuple

    _title: Optional[str]
    _xlabel: Optional[str]
    _ylabel: Optional[str]
    _zlabel: Optional[str]
    def __init__(
        self,
        object: Union[Output, Limbs, ArrayLike],
        *args,
        filter_kw: dict = {},
        show_path: bool = True,
        save: Optional[str] = None,
        close: bool = False,
        show: bool = False,
        save_kw: dict = {},
        **kwargs
    ):
        self._object = object
        self._output = None
        self._plotmethod = None
        self._ret = ()
        self._get_data(**filter_kw)
        self._get_labels()

        if self._plotmethod:
            kwargs.setdefault('method', self._plotmethod)

        ndim = len(self._array.shape)
        if ndim==1:
            self._edges = self._edges[0] if self._edges else None
            self._meshes = self._meshes[0] if self._meshes else None
            self._ret = plot_array_1d(
                self._array,
                self._edges,
                self._meshes,
                *args,
                plotter = self,
                **kwargs
            )
        elif ndim==2:
            colorbar = kwargs.pop('colorbar', {})
            if colorbar==True:
                colorbar={}
            if self._output and isinstance(colorbar, Mapping):
                colorbar.setdefault('label', self._output.labels.axis)
            self._ret = plot_array_2d(
                self._array,
                self._edges,
                self._meshes,
                *args,
                plotter = self,
                colorbar=colorbar,
                **kwargs
            )
        else:
            raise RuntimeError(f"Do not know how to plot {ndim}d")

        if self._output is not None:
            has_legend = 'label' in kwargs
            self.annotate_axes(show_path=show_path, legend=has_legend)

        if save:
            logger.log(SUBINFO, f'Write: {save}')
            savefig(save, **save_kw)
        if show: showfig()
        if close: closefig()

    def _get_output_data(self, *args, **kwargs):
        data = _mask_if_needed(self._output.data, *args, **kwargs)
        self._array = data
        self._edges = self._output.dd.edges_arrays
        self._meshes = self._output.dd.meshes_arrays

        self._plotmethod = self._output.labels.plotmethod
        if self._plotmethod in ('none', 'auto'):
            self._plotmethod = None

    def _get_array_data(self, *args, **kwargs):
        self._array = _mask_if_needed(self._object, *args, **kwargs)
        self._edges = ()
        self._meshes = ()

    def _get_data(self, *args, **kwargs):
        if isinstance(self._object, Output):
            self._output = self._object
        elif isinstance(self._object, Limbs):
            self._output = self._object.outputs[0]
        else:
            self._get_array_data(*args, **kwargs)
            return

        self._get_output_data(*args, **kwargs)

    def _get_labels(self):
        self._zlabel = None
        if not self._output:
            self._title = None
            self._xlabel = None
            self._ylabel = None
            return

        labels = self._output.labels

        self._title = labels.plottitle
        self._xlabel = self._output.dd.axis_label(0) or labels.xaxis or 'Index'

        self._ylabel = labels.axis
        if self._output.dd.dim==2:
            if self._plotmethod=='slicesx':
                self._xlabel = self._output.dd.axis_label(1) or labels.xaxis or 'Index'
                self._zlabel = self._output.dd.axis_label(0) or labels.xaxis or 'Index'
            elif self._plotmethod=='slicesy':
                self._zlabel = self._output.dd.axis_label(1) or labels.xaxis or 'Index'
            else:
                self._zlabel = self._ylabel
                self._ylabel = self._output.dd.axis_label(1)

    def annotate_axes(
        self,
        /,
        ax: Optional[Axes]=None,
        *,
        legend: bool=False,
        show_path: bool=True
    ) -> None:
        ax = ax or gca()
        labels = self._output.labels

        if self._title: ax.set_title(self._title)
        if self._xlabel: ax.set_xlabel(self._xlabel)
        if self._ylabel: ax.set_ylabel(self._ylabel)
        if self._zlabel:
            try:
                ax.set_zlabel(self._zlabel)
            except AttributeError:
                pass

        if legend:
            ax.legend()

        if show_path:
            path = labels.paths
            if not path:
                return

            fig = gcf()
            try:
                ax.text2D(0.05, 0.05, path[0], transform=fig.dpi_scale_trans)
            except AttributeError:
                ax.text(0.05, 0.05, path[0], transform=fig.dpi_scale_trans)

    @property
    def zlabel(self) -> Optional[str]:
        return self._zlabel

def plot_array_1d(
    array: NDArray,
    edges: Optional[NDArray],
    meshes: Optional[NDArray],
    *args,
    **kwargs
) -> Tuple[tuple, ...]:
    if edges is not None:
        return plot_array_1d_hist(array, edges, *args, **kwargs)
    elif meshes is not None:
        return plot_array_1d_vs(array, meshes, *args, **kwargs)
    else:
        return plot_array_1d_array(array, *args, **kwargs)

def plot_array_1d_hist(
    array: NDArray,
    edges: Optional[NDArray],
    *args,
    plotter: Optional[plot_auto]=None,
    **kwargs
) -> Tuple:
    return stairs(array, edges, *args, **kwargs)

def plot_array_1d_vs(
    array: NDArray,
    meshes: Optional[NDArray],
    *args,
    plotter: Optional[plot_auto]=None,
    **kwargs
) -> Tuple:
    return plot(meshes, array, *args, **kwargs)

def plot_array_1d_array(
    array: NDArray,
    *args,
    plotter: Optional[plot_auto]=None,
    **kwargs
) -> Tuple:
    return plot(array, *args, **kwargs)

def plot_output_1d(
    output: Output,
    *args,
    plotter: Optional[plot_auto]=None,
    **kwargs
) -> Tuple:
    array, edges, meshes = _get_data(output)
    return plot_array_1d(array, edges, meshes, *args, **kwargs)

def plot_output_1d_vs(
    output: Output,
    *args,
    plotter: Optional[plot_auto]=None,
    **kwargs
) -> Tuple:
    array, edges, _ = _get_data(output)
    return plot_array_1d_vs(array, edges, *args, **kwargs)

def plot_output_1d_meshes(
    output: Output,
    *args,
    plotter: Optional[plot_auto]=None,
    **kwargs
) -> Tuple:
    array, _, meshes = _get_data(output)
    return plot_array_1d_vs(array, meshes, *args, **kwargs)

def plot_output_1d_array(
    output: Output,
    *args,
    plotter: Optional[plot_auto]=None,
    **kwargs
) -> Tuple:
    return plot_array_1d_array(output.data, *args, **kwargs)

def plot_array_2d(
    array: NDArray,
    edges: EdgesLike,
    meshes: MeshesLike,
    *args,
    **kwargs
) -> Tuple[tuple, ...]:
    if edges:
        plot_array_2d_hist(array, edges, *args, **kwargs)
    elif meshes:
        plot_array_2d_vs(array, meshes, *args, **kwargs)
    else:
        plot_array_2d_array(array, *args, **kwargs)

def plot_array_2d_hist(
    dZ: NDArray,
    edges: List[NDArray],
    *args,
    method: Optional[str] = None,
    **kwargs
) -> Tuple:
    method = method in {'auto', None} and 'pcolormesh' or method
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
    meshes: List[NDArray],
    *args,
    method: Optional[str] = None,
    **kwargs
) -> Tuple:
    method = method in {'auto', None} and 'pcolormesh' or method
    fcn = {
            'surface': plot_array_2d_vs_surface,
            'wireframe': plot_array_2d_vs_wireframe,
            'pcolormesh': plot_array_2d_vs_pcolormesh,
            'pcolor': plot_array_2d_vs_pcolor,
            'slicesx': plot_array_2d_vs_slicesx,
            'slicesy': plot_array_2d_vs_slicesy,
            }.get(method, None)
    if fcn is None:
        raise RuntimeError("unimplemented")

    return fcn(array, meshes, *args, **kwargs)

def plot_array_2d_array(
    array: NDArray,
    *args,
    plotter: Optional[plot_auto]=None,
    **kwargs
) -> Tuple:
    kwargs.setdefault('aspect', 'auto')
    return plot_array_2d_hist_matshow(array, None, *args, **kwargs)

def plot_array_2d_hist_bar3d(
    dZ: NDArray,
    edges: List[NDArray],
    *args,
    plotter: Optional[plot_auto]=None,
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

def plot_array_2d_hist_pcolorfast(
    Z: NDArray,
    edges: List[NDArray],
    *args,
    plotter: Optional[plot_auto]=None,
    **kwargs
) -> Tuple:
    xedges, yedges = edges
    return pcolorfast(xedges, yedges, Z.T, *args, **kwargs)

def plot_array_2d_hist_pcolormesh(
    Z: NDArray,
    edges: List[NDArray],
    *args,
    plotter: Optional[plot_auto]=None,
    **kwargs
) -> Tuple:
    x, y = meshgrid(edges[0], edges[1], indexing='ij')
    return pcolormesh(x, y, Z, *args, **kwargs)

def plot_array_2d_hist_pcolor(
    Z: NDArray,
    edges: List[NDArray],
    *args,
    plotter: Optional[plot_auto]=None,
    **kwargs
) -> Tuple:
    x, y = meshgrid(edges[0], edges[1], indexing='ij')
    return pcolor(x, y, Z, *args, **kwargs)

def plot_array_2d_hist_imshow(
    Z: NDArray,
    edges: EdgesLike=None,
    *args,
    plotter: Optional[plot_auto]=None,
    **kwargs
):
    if edges:
        xedges, yedges = edges
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        kwargs.setdefault('extent', extent)
    kwargs.setdefault('origin', 'lower')
    return imshow(Z.T, *args, **kwargs)

def plot_array_2d_hist_matshow(
    Z: NDArray,
    edges: Optional[EdgesLike]=None,
    *args,
    plotter: Optional[plot_auto]=None,
    **kwargs
):
    kwargs.setdefault('fignum', False)
    if edges:
        xedges, yedges = edges
        extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]
        kwargs.setdefault('extent', extent)
    return matshow(Z.T, *args, **kwargs)

def plot_array_2d_vs_pcolormesh(
    Z: NDArray,
    meshes: List[NDArray],
    *args,
    plotter: Optional[plot_auto]=None,
    **kwargs
) -> Tuple:
    x, y = meshes
    kwargs.setdefault('shading', 'nearest')
    return pcolormesh(x, y, Z, *args, **kwargs)

def plot_array_2d_vs_pcolor(
    Z: NDArray,
    meshes: List[NDArray],
    *args,
    plotter: Optional[plot_auto]=None,
    **kwargs
) -> Tuple:
    x, y = meshes
    kwargs.setdefault('shading', 'nearest')
    return pcolormesh(x, y, Z, *args, **kwargs)

def plot_array_2d_vs_slicesx(
    Z: NDArray,
    meshes: List[NDArray],
    *args,
    plotter: Optional[plot_auto]=None,
    **kwargs
):
    x, y = meshes
    haslabels = False
    zlabel = plotter and plotter.zlabel or "value"

    for data, slicey, slicex in zip(Z, x, y):
        if (slicey[0]==slicey).all():
            label = f"slice {zlabel}={slicey[0]:.2g}"
            haslabels = True
        else:
            label = None
        plot(slicex, data, label=label)

    ax=gca()
    if haslabels is not None:
        ax.legend()

def plot_array_2d_vs_slicesy(
    Z: NDArray,
    meshes: List[NDArray],
    *args,
    **kwargs
):
    return plot_array_2d_vs_slicesx(Z.T, [mesh.T for mesh in reversed(meshes)], *args, **kwargs)

def plot_array_2d_vs_surface(
    Z: NDArray,
    meshes: List[NDArray],
    *args,
    plotter: Optional[plot_auto]=None,
    **kwargs
) -> Tuple:
    return plot_surface(meshes[0], meshes[1], Z, *args, **kwargs)

def plot_array_2d_vs_wireframe(
    Z: NDArray,
    meshes: List[NDArray],
    *args,
    # facecolors: Optional[str] = None,
    plotter: Optional[plot_auto]=None,
    cmap: Union[str, bool, None] = None,
    colorbar: Union[dict, bool] = False,
    **kwargs
) -> Tuple:
    X, Y = meshes

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

def _mask_if_needed(datain: ArrayLike, /, *, masked_value: Optional[float]=None) -> NDArray:
    data = asanyarray(datain)
    if masked_value is None:
        return data

    mask = (data==masked_value)
    return masked_array(data, mask=mask)

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
