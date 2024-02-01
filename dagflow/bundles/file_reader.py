from os import listdir
from pathlib import Path
from typing import TYPE_CHECKING, Any

from numpy import double, dtype, frombuffer, linspace
from numpy.typing import NDArray

from ..logger import SUBINFO, SUBSUBINFO, logger

if TYPE_CHECKING:
    import ROOT


file_readers = {}


class FileReaderMeta(type):
    """Metaclass for `FileReader` class, implementing `FileReader[file_name]` method"""

    _opened_files: dict[str, "FileReader"] = {}
    _last_used_file: str = ""

    def __init__(self, name: str, parents: tuple, args: dict):
        """Register the file reader based on the `_extension`"""

        super().__init__(name, parents, args)
        ext = args.get("_extension")
        if ext:
            file_readers[ext] = self

    def __getitem__(self, file_name: str | Path) -> "FileReader":
        file_name_str = file_name if isinstance(file_name, str) else str(file_name)
        try:
            ret = self._opened_files[file_name_str]
            action = "Read"
        except KeyError:
            ret = FileReader.open(file_name)
            action = "Use"

        logger.log(SUBINFO, f"{action}: {file_name_str}")
        self._opened_files[file_name_str] = ret
        self._last_used_file = file_name_str

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
            cls = file_readers[ext]
        except KeyError:
            raise ValueError(
                f"Do not know how to load ext {ext}. Available file_readers:"
                f" {', '.join(file_readers)}"
            )

        try:
            return cls(file_name)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Can not open file {file_name!s} (loader {ext})") from e

    @classmethod
    def release_files(cls):
        for k, v in list(cls._opened_files.items()):
            v._close()
            del cls._opened_files[k]

    def _close(self):
        pass

    def keys(self) -> tuple[str, ...]:
        raise RuntimeError("not implemented method")

    def _get_object_impl(self, object_name: str) -> Any:
        raise RuntimeError("not implemented method")

    def _get_object(self, object_name: str) -> Any:
        try:
            return self._get_object_impl(object_name)
        except KeyError as e:
            raise KeyError(f"Can not read {object_name} from {self._file_name!s}") from e

    def _get_graph(self, object_name: str) -> tuple[NDArray, NDArray]:
        raise RuntimeError("not implemented method")

    def get_graph(self, object_name: str) -> tuple[NDArray, NDArray]:
        x, y = self._get_graph(object_name)
        logger.log(SUBSUBINFO, f"graph: x {x[0]}→{x[-1]}, ymin={y.min()}, ymax={y.max()}")
        return x, y

    def _get_hist(self, object_name: str) -> tuple[NDArray, NDArray]:
        raise RuntimeError("not implemented method")

    def get_hist(self, object_name: str) -> tuple[NDArray, NDArray]:
        x, y = self._get_hist(object_name)
        logger.log(
            SUBSUBINFO, f"hist: x {x[0]}→{x[-1]}, hmin={y.min()}, hmax={y.max()}, hsum={y.sum()}"
        )
        return x, y

    def _get_array(self, object_name: str) -> NDArray:
        raise RuntimeError("not implemented method")

    def get_array(self, object_name: str) -> NDArray:
        a = self._get_array(object_name)
        logger.log(
            SUBSUBINFO,
            f"array {'x'.join(map(str,a.shape))}: min={a.min()}, max={a.max()}, sum={a.sum()}",
        )
        return a


class FileReaderArray(FileReader):
    def _get_xy(self, object_name: str) -> tuple[NDArray, NDArray]:
        raise RuntimeError("not implemented method")

    def _get_graph(self, object_name: str) -> tuple[NDArray, NDArray]:
        return self._get_xy(object_name)

    def _get_hist(self, object_name: str) -> tuple[NDArray, NDArray]:
        x, y = self._get_xy(object_name)
        return x, y

    def _get_array(self, object_name: str) -> NDArray:
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

    def keys(self) -> tuple[str, ...]:
        return tuple(self._file.keys())

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

    def keys(self) -> tuple[str, ...]:
        return tuple(self._file.keys())

    def get_xy(self, object_name: str) -> tuple[NDArray, NDArray]:
        data = self._get_object(object_name)
        cols = data.dtype.names
        return data[cols[0]], data[cols[1]]

    def _get_array(self, object_name: str) -> NDArray:
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

    def keys(self) -> tuple[str, ...]:
        return tuple(file for file in listdir(self._file_name) if file.endswith(self._extension))

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

        def _get_hist(self, object_name: str) -> tuple[NDArray, NDArray]:
            import ROOT

            obj = self._get_object(object_name)
            if isinstance(obj, ROOT.TH1) and obj.GetDimension() == 1:
                return _get_bin_edges(obj.GetXaxis()), _get_buffer_hist1(obj, flows=False)

            raise ValueError(f"Do not know ho to convert {obj} to hist")

        def _get_graph(self, object_name: str) -> tuple[NDArray, NDArray]:
            import ROOT

            obj = self._get_object(object_name)

            if isinstance(obj, ROOT.TH1) and obj.GetDimension() == 1:
                return _get_bin_left_edges(obj.GetXaxis()), _get_buffer_hist1(obj, flows=False)
            if isinstance(obj, ROOT.TGraph):
                return _get_buffers_graph(obj)

            raise ValueError(f"Do not know ho to convert {obj} to graph")

        def _get_array(self, object_name: str) -> NDArray:
            import ROOT

            obj = self._get_object(object_name)

            if isinstance(obj, ROOT.TH1) and obj.GetDimension() == 1:
                return _get_buffer_hist1(obj)
            if isinstance(obj, ROOT.TH2) and obj.GetDimension() == 2:
                return _get_buffer_hist2(obj)
            if isinstance(obj, (ROOT.TMatrixD, ROOT.TMatrixF)):
                return _get_buffer_matrix(obj)

            raise ValueError(f"Do not know ho to convert {obj} to array")

        def keys(self) -> tuple[str, ...]:
            return tuple(key.split(';', 1)[0] for key in self._file.GetListOfKeys())

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

        def _get_hist(self, object_name: str) -> tuple[NDArray, NDArray]:
            obj = self._get_object_impl(object_name)
            y, x = obj.to_numpy()
            return x, y

        def _get_graph(self, object_name: str) -> tuple[NDArray, NDArray]:
            obj = self._get_object_impl(object_name)
            y, x = obj.to_numpy()
            return x[:-1], y

        def _get_array(self, object_name: str) -> NDArray:
            obj = self._get_object_impl(object_name)
            y, _ = obj.to_numpy()
            return y

        def keys(self) -> tuple[str, ...]:
            return tuple(self._file.keys())


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
