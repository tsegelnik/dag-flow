from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import allclose, array, dtype, frombuffer
from numpy.typing import NDArray

from nestedmapping.visitor import NestedMappingVisitor

from ..core.output import Output
from ..tools.logger import INFO1, INFO2, logger

if TYPE_CHECKING:
    from contextlib import suppress
    from pathlib import Path
    from typing import Any

    with suppress(ImportError):
        from ROOT import TH1, TH1D, TH2, TH2D, TDirectory, TFile, TObject


def to_root(output: Output | Any) -> dict[str, TObject]:
    rets = {}
    if not isinstance(output, Output):
        return rets

    dd = output.dd
    dim = dd.dim
    if dim == 1:
        if dd.axes_edges:
            rets["hist"] = to_TH1(output)
        if dd.axes_meshes:
            rets["graph"] = to_TGraph(output)
    elif dim == 2:
        if dd.axes_edges:
            rets["hist"] = to_TH2(output)
        if dd.axes_meshes:
            rets["graph"] = to_TGraph2(output)
    else:
        raise RuntimeError(f"Unsupported output dimension: {dim}")

    return rets


def _buffer_clean(buf: NDArray) -> NDArray:
    # Should use ascontiguousarray, but it does not work properly for 'f'
    return array(buf, dtype="d")


def edges_to_args(edges: NDArray, *, rtol: float = 1.0e-9) -> tuple:
    edges = _buffer_clean(edges)
    widths = edges[1:] - edges[:-1]
    if allclose(widths, widths.max(), rtol=rtol, atol=0):
        return edges.size - 1, edges[0], edges[-1]

    return edges.size - 1, edges


def get_buffer_hist1(hist: TH1) -> NDArray:
    buf = hist.GetArray()
    buf = frombuffer(buf, dtype(buf.typecode), hist.GetNbinsX() + 2)
    return buf[1:-1]


def get_buffer_hist2(hist: TH2) -> NDArray:
    nx, ny = hist.GetNbinsX(), hist.GetNbinsY()
    buf = hist.GetArray()
    res = frombuffer(buf, dtype(buf.typecode), (nx + 2) * (ny + 2)).reshape((ny + 2, nx + 2))
    # TODO: check order
    return res[1 : ny + 1, 1 : nx + 1].T


def to_TH1(output: Output) -> TH1D:
    from ROOT import TH1D

    data = _buffer_clean(output.data)
    labels = output.labels

    edges = edges_to_args(output.dd.axes_edges[0].data)
    hist = TH1D("", labels.roottitle, *edges)
    buffer = get_buffer_hist1(hist)

    buffer[:] = data
    hist.SetEntries(data.sum())
    hist.SetXTitle(output.dd.axes_edges[0].labels.rootaxis)
    hist.SetYTitle(labels.rootaxis)

    return hist


def to_TH2(output: Output) -> TH2D:
    from ROOT import TH2D

    data = _buffer_clean(output.data)
    labels = output.labels

    edgesX = edges_to_args(output.dd.axes_edges[0].data)
    edgesY = edges_to_args(output.dd.axes_edges[1].data)
    hist = TH2D("", labels.roottitle, *(edgesX + edgesY))
    buffer = get_buffer_hist2(hist)

    buffer[:] = data
    hist.SetEntries(data.sum())
    hist.SetXTitle(output.dd.axes_edges[0].labels.rootaxis)
    hist.SetYTitle(output.dd.axes_edges[1].labels.rootaxis)
    hist.SetZTitle(labels.rootaxis)

    return hist


def to_TGraph(output):
    from ROOT import TGraph

    labels = output.labels

    x = _buffer_clean(output.dd.axes_meshes[0].data)
    y = _buffer_clean(output.data)

    title = labels.roottitle
    xtitle = output.dd.axes_meshes[0].labels.rootaxis
    ytitle = labels.rootaxis

    graph = TGraph(x.size, x, y)
    graph.SetTitle(f"{title};{xtitle};{ytitle}")

    return graph


def to_TGraph2(output):
    from ROOT import TGraph2D

    labels = output.labels

    x = _buffer_clean(output.dd.axes_meshes[0].data)
    y = _buffer_clean(output.dd.axes_meshes[1].data)
    z = _buffer_clean(output.data)

    title = labels.roottitle
    xtitle = output.dd.axes_meshes[0].labels.rootaxis
    ytitle = output.dd.axes_meshes[1].labels.rootaxis
    ztitle = labels.rootaxis

    graph = TGraph2D(x.size, x.ravel(), y.ravel(), z.ravel())
    graph.SetTitle(f"{title};{xtitle};{ytitle};{ztitle}")

    return graph


class ExportToRootVisitor(NestedMappingVisitor):
    __slots__ = ("_file", "_cwd", "_prevd", "_level")
    _file: TFile
    _prevd: list["TDirectory"]
    _cwd: TDirectory
    _level: int

    def __init__(self, filename: "Path" | str):
        from ROOT import TFile

        filename = str(filename)
        logger.log(INFO1, f"Create {filename}")
        self._file = TFile(filename, "RECREATE")
        self._file.AddDirectory(False)
        if self._file.IsZombie():
            raise RuntimeError(f"File {filename!s} is zombie")
        self._prevd = []
        self._cwd = self._file
        self._level = 0

    def start(self, dct):
        pass

    def enterdict(self, key, v):
        if not key:
            return

        self._level += 1

        cwd = self._file
        for subkey in key:
            cwd = cwd.mkdir(subkey, "", True)

        self._prevd.append(self._cwd)
        self._cwd = cwd

    def visit(self, key, value):
        path = "/".join(key[self._level :])

        objects = to_root(value)
        if not objects:
            return

        name = path
        hist = objects.pop("hist", None)
        if hist is not None:
            logger.log(INFO2, f"write {'/'.join(key)}")
            self._cwd.WriteTObject(hist, name, "overwrite")
            name = f"{name}_graph"

        graph = objects.pop("graph", None)
        if graph is not None:
            logger.log(INFO2, f"write {'/'.join(key)}")
            self._cwd.WriteTObject(graph, name, "overwrite")

        if objects:
            raise RuntimeError(f"Unsupported ROOT objects: {objects}")

    def exitdict(self, k, v):
        if not self._prevd:
            return

        self._level -= 1
        self._cwd = self._prevd.pop()

    def stop(self, dct):
        logger.log(INFO1, f"Close {self._file.GetName()}")
        self._file.Close()
