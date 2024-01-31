from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from os.path import basename
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from numpy import allclose, double, dtype, frombuffer, linspace
from numpy.typing import NDArray
from schema import And
from schema import Optional as SchemaOptional
from schema import Or, Schema, Use

from multikeydict.typing import TupleKey

from ..lib.Array import Array
from ..logger import SUBINFO, logger
from ..storage import NodeStorage
from ..tools.schema import (
    AllFileswithExt,
    IsFilenameSeqOrFilename,
    IsReadable,
    IsStrSeqOrStr,
    LoadFileWithExt,
    LoadYaml,
)

if TYPE_CHECKING:
    import ROOT


_loaders = {}


class FileReaderMeta(type):
    """Metaclass for `FileReader` class, implementing `FileReader[file_name]` method"""

    _opened_files: dict[str, "FileReader"] = {}
    _last_used_file: str = ""

    def __init__(self, name: str, parents: tuple, args: dict):
        """Register the file reader based on the `_extension`"""

        super().__init__(name, parents, args)
        ext = args["_extension"]
        if ext:
            _loaders[ext] = self

    def __getitem__(self, file_name: str | Path) -> "FileReader":
        action = "Read"
        self._last_used_file = (
            file_name_str := file_name if isinstance(file_name, str) else str(file_name)
        )
        try:
            ret = self._opened_files[file_name_str]
        except KeyError:
            action = "Use"
            ret = FileReader.open(file_name)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Can not open file {file_name_str}") from e

        logger.log(SUBINFO, f"{action}: {file_name_str}")
        self._opened_files[file_name_str] = ret

        return ret


class FileReader(metaclass=FileReaderMeta):
    _extension: str = ""
    _file: Any = None
    _file_name: Path = Path("")
    _opened_files: dict[str, "FileReader"] = FileReaderMeta._opened_files

    def __init__(self, file_name: str | Path):
        self._file_name = Path(file_name)

    @classmethod
    def open(cls, file_name: str | Path) -> "FileReader":
        if not isinstance(cls, FileReader):
            return cls(file_name)

        file_path = file_name if isinstance(file_name, Path) else Path(file_name)
        ext = file_path.suffix()  # pyright: ignore [reportGeneralTypeIssues]

        try:
            cls = _loaders[ext]
        except KeyError:
            raise RuntimeError(
                f"Do not know how to load ext {ext}. Available loaders: {', '.join(_loaders)}"
            )

        return cls(file_name)

    @classmethod
    def release_files(cls):
        for k, v in list(cls._opened_files.items()):
            v._close()
            del cls._opened_files[k]

    def _close(self):
        pass

    def _get_object_impl(self, object_name: str) -> Any:
        raise RuntimeError("not implemented method")

    def _get_object(self, object_name: str) -> Any:
        try:
            return self._get_object_impl(object_name)
        except KeyError as e:
            raise KeyError(f"Can not read {object_name} from {self._file_name!s}") from e

    def get_graph(self, object_name: str) -> tuple[NDArray, NDArray]:
        raise RuntimeError("not implemented method")

    def get_hist(self, object_name: str) -> tuple[NDArray, NDArray]:
        raise RuntimeError("not implemented method")

    def get_array(self, object_name: str) -> NDArray:
        raise RuntimeError("not implemented method")


class FileReaderArray(FileReader):
    def _get_xy(self, object_name: str) -> tuple[NDArray, NDArray]:
        raise RuntimeError("not implemented method")

    def get_graph(self, object_name: str) -> tuple[NDArray, NDArray]:
        return self._get_xy(object_name)

    def get_hist(self, object_name: str) -> tuple[NDArray, NDArray]:
        x, y = self._get_xy(object_name)
        return x, y

    def get_array(self, object_name: str) -> NDArray:
        return self._get_object(object_name)


class FileReaderNPZ(FileReaderArray):
    _extension: str = ".npz"

    def __init__(self, file_name: str | Path):
        super().__init__(file_name)
        from numpy import load

        self._file = load(self._file_name)

    def _close(self):
        del self._file

    def _get_object_impl(self, object_name: str) -> Any:
        return self._file[object_name]

    def get_xy(self, object_name: str) -> tuple[NDArray, NDArray]:
        data = self._get_object(object_name)
        cols = data.dtype.names
        return data[cols[0]], data[cols[1]]


class FileReaderHDF5(FileReaderArray):
    _extension: str = ".hdf5"

    def __init__(self, file_name: str | Path):
        super().__init__(file_name)
        from h5py import File

        self._file = File(self._file_name, "r")

    def _close(self):
        self._file.close()

    def _get_object_impl(self, object_name: str) -> Any:
        return self._file[object_name]

    def get_xy(self, object_name: str) -> tuple[NDArray, NDArray]:
        data = self._get_object(object_name)
        cols = data.dtype.names
        return data[cols[0]], data[cols[1]]

    def get_array(self, object_name: str) -> tuple[NDArray, NDArray]:
        ret = self._get_object(object_name)
        return ret[:]


class FileReaderTSV(FileReaderArray):
    _extension: str = ".tsv"

    def __init__(self, file_name: str | Path):
        super().__init__(file_name)
        if not self._file_name.is_dir():
            raise FileNotFoundError(file_name)

    def _get_object_impl(self, object_name: str) -> Any:
        from numpy import loadtxt

        filename = self._file_name / f"{object_name}.{self._extension}"
        return loadtxt(filename, unpack=True)

    def get_xy(self, object_name: str) -> tuple[NDArray, NDArray]:
        data = self._get_object(object_name)
        return data[0], data[1]


try:
    import ROOT
except ImportError:

    class FileReaderROOTROOT(FileReader):
        _extension: str = ".root"

        def __init__(self, file_name: str | Path):
            super().__init__(file_name)
            from ROOT import TFile

            self._file = TFile(file_name)
            if self._file.IsZombie():
                raise FileNotFoundError(file_name)

        def _close(self):
            self._file.Close()

        def _get_object_impl(self, object_name: str) -> Any:
            ret = self._file.Get(object_name)
            if not ret:
                raise KeyError(object_name)
            return ret

        def get_hist(self, object_name: str) -> tuple[NDArray, NDArray]:
            import ROOT

            obj = self._get_object(object_name)
            if isinstance(obj, ROOT.TH1) and obj.GetDimension() == 1:
                return _get_bin_edges(obj.GetXaxis()), _get_buffer_hist1(obj, flows=False)

            raise ValueError(f"Do not know ho to convert {obj} to hist")

        def get_graph(self, object_name: str) -> tuple[NDArray, NDArray]:
            import ROOT

            obj = self._get_object(object_name)

            if isinstance(obj, ROOT.TH1) and obj.GetDimension() == 1:
                return _get_bin_left_edges(obj.GetXaxis()), _get_buffer_hist1(obj, flows=False)
            if isinstance(obj, ROOT.TGraph):
                return _get_buffers_graph(obj)

            raise ValueError(f"Do not know ho to convert {obj} to graph")

        def get_array(self, object_name: str) -> NDArray:
            import ROOT

            obj = self._get_object(object_name)

            if isinstance(obj, ROOT.TH1) and obj.GetDimension() == 1:
                return _get_buffer_hist1(obj)
            if isinstance(obj, ROOT.TH2) and obj.GetDimension() == 2:
                return _get_buffer_hist2(obj)
            if isinstance(obj, (ROOT.TMatrixD, ROOT.TMatrixF)):
                return _get_buffer_matrix(obj)

            raise ValueError(f"Do not know ho to convert {obj} to array")

else:

    class FileReaderROOTUpROOT(FileReader):
        _extension: str = ".root"

        def __init__(self, file_name: str | Path):
            super().__init__(file_name)
            from uproot import open

            self._file = open(file_name)

        def _close(self):
            self._file.close()

        def _get_object_impl(self, object_name: str) -> Any:
            return self._file[object_name]

        def get_hist(self, object_name: str) -> tuple[NDArray, NDArray]:
            obj = self._get_object_impl(object_name)
            y, x = obj.to_numpy()
            return x, y

        def get_graph(self, object_name: str) -> tuple[NDArray, NDArray]:
            obj = self._get_object_impl(object_name)
            y, x = obj.to_numpy()
            return x[:-1], y

        def get_array(self, object_name: str) -> NDArray:
            obj = self._get_object_impl(object_name)
            y, _ = obj.to_numpy()
            return y

_schema_cfg = Schema(
    {
        "name": str,
        "filenames": And(IsFilenameSeqOrFilename, AllFileswithExt(*_loaders.keys())),
        SchemaOptional("merge_x", default=False): bool,
        SchemaOptional("x", default="x"): str,
        SchemaOptional("y", default="y"): str,
        SchemaOptional("normalize", default=False): bool,
        SchemaOptional("replicate", default=((),)): Or((IsStrSeqOrStr,), [IsStrSeqOrStr]),
        SchemaOptional("objects", default=lambda s: s): Or(
            Callable, And({str: str}, Use(lambda dct: lambda s: dct.get(s, s)))
        ),
    }
)

_schema_loadable_cfg = And(
    {"load": Or(str, And(Path, Use(str))), SchemaOptional(str): object},
    Use(
        LoadFileWithExt(yaml=LoadYaml, key="load", update=True),
        error="Failed to load {}",
    ),
    _schema_cfg,
)


def _validate_cfg(cfg):
    if isinstance(cfg, dict) and "load" in cfg:
        return _schema_loadable_cfg.validate(cfg)
    else:
        return _schema_cfg.validate(cfg)


def get_filename(
    filenames: Sequence[str],
    key: TupleKey,
    *,
    single_key: bool = False,
    multiple_files: bool = False,
) -> str:
    if (single_key or not multiple_files) and len(filenames) == 1:
        return filenames[0]
    checked_filenames = []
    skey = "_".join(key)
    for filename in filenames:
        if Path(filename).is_dir():
            if filename.endswith(".tsv"):
                ext = filename[-3:]
                bname = basename(filename[:-4])
                ifilename = f"{filename}/{bname}_{skey}.{ext}"
                checked_filenames.append(ifilename)
                if IsReadable(ifilename):
                    return ifilename
        elif "{key}" in filename:
            ifilename = filename.format(key=skey)
            checked_filenames.append(ifilename)
            if IsReadable(ifilename):
                return ifilename
        elif all(subkey in filename for subkey in key):
            checked_filenames.append(filename)
            if IsReadable(filename):
                return filename

    raise RuntimeError(f"Unable to find readable filename for {key}. Checked: {checked_filenames}")


def load_hist(acfg: Optional[Mapping] = None, **kwargs):
    acfg = dict(acfg or {}, **kwargs)
    cfg = _validate_cfg(acfg)

    name = cfg["name"]
    filenames = cfg["filenames"]
    keys = cfg["replicate"]
    objects = cfg["objects"]
    normalize = cfg["normalize"]

    xname = name, cfg["x"]
    yname = name, cfg["y"]

    try:
        ext = next(ext for ext in _loaders if filenames[0].endswith(f".{ext}"))
    except StopIteration:
        raise RuntimeError(f"Unable to find extension for: {filenames[0]}")

    loader, multiple_files = _loaders.get(ext)
    if loader is None:
        raise RuntimeError(f"Unable to find loader for: {filenames[0]}")
    single_key = len(keys) == 1

    edges_list = []
    data = {}
    for key in keys:
        filename = get_filename(
            filenames, key, single_key=single_key, multiple_files=multiple_files
        )

        skey = ".".join(key)
        iname = objects(skey)
        x, y = loader(filename, iname)
        logger.log(SUBINFO, f"Read: {filename}")

        if normalize:
            y /= y.sum()

        data[key] = x, y
        edges_list.append(x)

    if cfg["merge_x"]:
        x0 = edges_list[0]
        for xi in edges_list[1:]:
            if not allclose(x0, xi, atol=0, rtol=0):
                raise RuntimeError("load_graph: inconsistent x axes, unable to merge.")

        commonedges, _ = Array.make_stored(".".join(xname), x0)
    else:
        commonedges = None

    storage = NodeStorage(default_containers=True)
    with storage:
        for key, (x, y) in data.items():
            if commonedges:
                edges = commonedges
            else:
                xkey = ".".join(xname + key)
                edges, _ = Array.make_stored(xkey, x)
            ykey = ".".join(yname + key)
            Array.make_stored(ykey, y, edges=edges)

    NodeStorage.update_current(storage, strict=True)

    return storage


def _get_buffer_hist1(h: "ROOT.TH1", flows: bool = False) -> NDArray:
    """Return TH1* histogram data buffer
    if flows=False, exclude underflow and overflow
    """
    buf = h.GetArray()
    buf = frombuffer(buf, dtype(buf.typecode), h.GetNbinsX() + 2)
    if not flows:
        buf = buf[1:-1]

    return buf.copy()


def _get_buffer_hist2(h, flows=False):
    """Return histogram data buffer
    if flows=False, exclude underflow and overflow
    NOTE: buf[biny][binx] is the right access signature
    """
    nx, ny = h.GetNbinsX(), h.GetNbinsY()
    buf = h.GetArray()
    res = frombuffer(buf, dtype(buf.typecode), (nx + 2) * (ny + 2)).reshape((ny + 2, nx + 2))
    if not flows:
        res = res[1 : ny + 1, 1 : nx + 1]

    return res.copy()


def _get_bin_edges(ax: "ROOT.TAxis") -> NDArray:
    """Get the array with bin edges"""
    xbins = ax.GetXbins()
    n = xbins.GetSize()
    if n > 0:
        lims = frombuffer(xbins.GetArray(), double, n)
        return lims.copy()
    return linspace(ax.GetXmin(), ax.GetXmax(), ax.GetNbins() + 1)


def _get_bin_left_edges(ax: "ROOT.TAxis") -> NDArray:
    """Get the array with bin left edges"""
    xbins = ax.GetXbins()
    n = xbins.GetSize()
    if n > 0:
        lims = frombuffer(xbins.GetArray(), double, n)
        return lims[:-1].copy()
    return linspace(ax.GetXmin(), ax.GetXmax(), ax.GetNbins() + 1)[:-1]


def _get_buffers_graph(g: "ROOT.TGraph") -> tuple[NDArray, NDArray]:
    """Get TGraph x and y buffers"""
    npoints = g.GetN()
    if npoints == 0:
        raise RuntimeError("Got graph with 0 points")

    return (
        frombuffer(g.GetX(), dtype=double, count=npoints).copy(),
        frombuffer(g.GetY(), dtype=double, count=npoints).copy(),
    )


def _get_buffer_matrix(m):
    """Get TMatrix buffer"""
    cbuf = m.GetMatrixArray()
    res = frombuffer(cbuf, dtype(cbuf.typecode), m.GetNoElements()).reshape(
        m.GetNrows(), m.GetNcols()
    )
    return res.astype(double).copy()
