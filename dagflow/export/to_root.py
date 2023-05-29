from multikeydict.visitor import NestedMKDictVisitor

from numpy import allclose, frombuffer, dtype

from typing import List, Dict, Union, Any, Tuple, TYPE_CHECKING
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pathlib import Path
    from ..output import Output

    try:
        from ROOT import TFile, TDirectory, TObject, TH1D, TH2D, TH1, TH2
    except ImportError:
        pass

def to_root(output: Union['Output', Any]) -> Dict[str, 'TObject']:
    rets = {}
    if not isinstance(output, 'Output'):
        return rets

    dd = output.dd
    dim = dd.dim
    if dim==1:
        if dd.axes_edges:
            rets['hist'] = to_TH1(output)
        if dd.axes_nodes:
            rets['graph'] = to_TGraph(output)
    elif dim==2:
        if dd.axes_edges:
            rets['hist'] = to_TH2(output)
        if dd.axes_nodes:
            rets['graph'] = to_TGraph2(output)
    else:
        raise RuntimeError('Unsupported output dimension: {dim}')

    return rets

def edges_to_args(edges: NDArray, *, rtol: float=1.e-9) -> Tuple:
    widths = edges[1:]-edges[:-1]
    if allclose(widths, widths.max(), rtol=rtol, atol=0):
        return edges.size-1, edges[0], edges[-1]

    return edges.size-1, edges.astype('d', copy=False)

def get_buffer_hist1(hist: 'TH1') -> NDArray:
    buf = hist.GetArray()
    buf = frombuffer(buf, dtype(buf.typecode), h.GetNbinsX()+2)
    return buf[1:-1]

def get_buffer_hist2(hist: 'TH2') -> NDArray:
    nx, ny = hist.GetNbinsX(), hist.GetNbinsY()
    buf = hist.GetArray()
    res = frombuffer(buf, dtype(buf.typecode), (nx+2)*(ny+2)).reshape((ny+2, nx+2))
    # TODO: check order
    return res[1:ny+1, 1:nx+1]

def to_TH1(output: "Output") -> "TH1D":
    from ROOT import TH1D

    data = output.data
    labels = output.labels

    edges = edges_to_args(output.dd.axes_edges[0])
    hist = TH1D("", labels.roottitle, *edges)
    buffer = get_buffer_hist1(hist)

    buffer[:] = data
    hist.SetEntries(data.sum())
    hist.SetXTitle(output.dd.axis_label(0, 'edges', root=True),)
    hist.SetYTitle(labels.rootaxis)

    return hist

def to_TH2(output: "Output") -> "TH2D":
    from ROOT import TH2D

    data = output.data
    labels = output.labels

    edgesX = edges_to_args(output.dd.axes_edges[0])
    edgesY = edges_to_args(output.dd.axes_edges[1])
    hist = TH1D("", labels.roottitle, *(edgesX+edgesY))
    buffer = get_buffer_hist2(hist)

    buffer[:] = data
    hist.SetEntries(data.sum())
    hist.SetXTitle(output.dd.axis_label(0, 'edges', root=True))
    hist.SetYTitle(output.dd.axis_label(1, 'edges', root=True))
    hist.SetZTitle(labels.rootaxis)

    return hist

def to_TGraph(output):
    pass

def to_TGraph2(output):
    pass

class ExportToRootVisitor(NestedMKDictVisitor):
    __slots__ = ('_file', '_cwd', '_prevd')
    _file: 'TFile'
    _prevd: List["TDirectory"]
    _cwd: "TDirectory"

    def __init__(self, filename: Union[Path,str]):
        from ROOT import TFile
        self._file = TFile(str(filename))
        if self._file.IsZombie():
            raise RuntimeError(f'File {filename!s} is zombie')
        self._prevd = []
        self._cwd = self._file

    def start(self, dct):
        pass

    def enterdict(self, key, v):
        path = '/'.join(key)
        self._cwd = self._cwd.mkdir(path)
        self._prevd.append(self._cwd)

    def visit(self, key, value):
        path = '/'.join(key)

        objects = to_root(value)
        if not objects:
            return

        name = path
        hist = objects.pop('hist', None)
        if hist is not None:
            self._cwd.WriteTObject(hist, name, 'overwrite')
            name = f'{name}_graph'

        graph = objects.pop('graph', None)
        if graph is not None:
            self._cwd.WriteTObject(graph, name, 'overwrite')

        if objects:
            raise RuntimeError(f'Unsupported ROOT objects: {objects}')

    def exitdict(self, k, v):
        self._cwd = self._prevd.pop()

    def stop(self, dct):
        pass

try:
    import ROOT
except ImportError as e:
    class ExportToRootVisitor(NestedMKDictVisitor):
        def __init__(self, *args, **kwargs):
            raise ImportError("ROOT") from e
