from collections.abc import Sequence
from contextlib import suppress
from os.path import basename
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Optional, Tuple

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

_extensions = {"root", "hdf5", "tsv", "txt", "npz"}
_schema_cfg = Schema(
    {
        "name": str,
        "filenames": And(IsFilenameSeqOrFilename, AllFileswithExt(*_extensions)),
        SchemaOptional("merge_x", default=False): bool,
        SchemaOptional("x", default="x"): str,
        SchemaOptional("y", default="y"): str,
        SchemaOptional("replicate", default=((),)): Or((IsStrSeqOrStr,), [IsStrSeqOrStr]),
        SchemaOptional("objects", default={}): {str: str},
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


def load_graph(acfg: Optional[Mapping] = None, **kwargs):
    acfg = dict(acfg or {}, **kwargs)
    cfg = _validate_cfg(acfg)

    name = cfg["name"]
    filenames = cfg["filenames"]
    keys = cfg["replicate"]
    objects = cfg["objects"]

    xname = name, cfg["x"]
    yname = name, cfg["y"]

    try:
        ext = next(ext for ext in _extensions if filenames[0].endswith(f".{ext}"))
    except StopIteration:
        raise RuntimeError(f"Unable to find extension for: {filenames[0]}")

    loader, multiple_files = _loaders.get(ext)
    if loader is None:
        raise RuntimeError(f"Unable to find loader for: {filenames[0]}")
    single_key = len(keys) == 1

    meshes = []
    data = {}
    for key in keys:
        filename = get_filename(
            filenames, key, single_key=single_key, multiple_files=multiple_files
        )

        skey = ".".join(key)
        iname = objects.get(skey, skey)
        x, y = loader(filename, iname)
        logger.log(SUBINFO, f"Read: {filename}")
        data[key] = x, y
        meshes.append(x)

    if cfg["merge_x"]:
        x0 = meshes[0]
        for xi in meshes[1:]:
            if not allclose(x0, xi, atol=0, rtol=0):
                raise RuntimeError("load_graph: inconsistent x axes, unable to merge.")

        commonmesh, _ = Array.make_stored(".".join(xname), x0)
    else:
        commonmesh = None

    storage = NodeStorage(default_containers=True)
    with storage:
        for key, (x, y) in data.items():
            if commonmesh:
                mesh = commonmesh
            else:
                xkey = ".".join(xname + key)
                mesh, _ = Array.make_stored(xkey, x)
            ykey = ".".join(yname + key)
            Array.make_stored(ykey, y, meshes=mesh)

    NodeStorage.update_current(storage, strict=True)

    return storage


def _load_tsv(filename: str, name: str) -> NDArray:
    from numpy import loadtxt

    return loadtxt(filename, unpack=True)


def _load_hdf5(filename: str, name: str) -> Tuple[NDArray, NDArray]:
    from h5py import File

    file = File(filename, "r")
    try:
        data = file[name]
    except KeyError as e:
        raise RuntimeError(f"Unable to read {name} from {filename}") from e

    cols = data.dtype.names
    return data[cols[0]], data[cols[1]]


def _load_npz(filename: str, name: str) -> Tuple[NDArray, NDArray]:
    from numpy import load

    file = load(filename)
    try:
        data = file[name]
    except KeyError as e:
        raise RuntimeError(f"Unable to read {name} from {filename}") from e

    cols = data.dtype.names
    return data[cols[0]], data[cols[1]]


def _load_root_uproot(filename: str, name: str) -> Tuple[NDArray, NDArray]:
    if not name:
        raise RuntimeError(f"Need an object name to read from {filename}")
    from uproot import open

    file = open(filename)
    try:
        data = file[name]
    except KeyError as e:
        raise RuntimeError(f"Unable to read {name} from {filename}") from e

    y, x = data.to_numpy()

    return x[:-1], y


def _load_root_ROOT(filename: str, name: str) -> Tuple[NDArray, NDArray]:
    if not name:
        raise RuntimeError(f"Need an object name to read from {filename}")
    from ROOT import TFile

    file = TFile(filename)
    if file.IsZombie():
        raise RuntimeError(f"Can not open file {filename}")
    data = file.Get(name)
    if not data:
        raise RuntimeError(f"Unable to read {name} from {filename}")

    return _get_buffers(data)


def _load_root(filename: str, *args, **kwargs) -> Tuple[NDArray, NDArray]:
    with suppress(AttributeError):
        return _load_root_uproot(filename, *args, **kwargs)

    try:
        return _load_root_ROOT(filename, *args, **kwargs)
    except ImportError:
        raise RuntimeError(
            f"Error reading file {filename}. `uproot` is unable, `ROOT` is not found."
        )


_loaders = {
    "txt": (_load_tsv, True),
    "tsv": (_load_tsv, True),
    "root": (_load_root, False),
    "npz": (_load_npz, False),
    "hdf5": (_load_hdf5, False),
}


def _get_buffer_hist1(h: "ROOT.TH1", flows: bool = False) -> NDArray:
    """Return TH1* histogram data buffer
    if flows=False, exclude underflow and overflow
    """
    buf = h.GetArray()
    buf = frombuffer(buf, dtype(buf.typecode), h.GetNbinsX() + 2)
    if not flows:
        buf = buf[1:-1]

    return buf.copy()


def _get_bin_left_edges(ax: "ROOT.TAxis") -> NDArray:
    """Get the array with bin left edges"""
    xbins = ax.GetXbins()
    n = xbins.GetSize()
    if n > 0:
        lims = frombuffer(xbins.GetArray(), double, n)
        return lims[:-1].copy()
    return linspace(ax.GetXmin(), ax.GetXmax(), ax.GetNbins() + 1)[:-1]


def _get_buffers_hist1(h: "ROOT.TH1") -> Tuple[NDArray, NDArray]:
    """Get X(left edges)/Y buffers of 1D histogram"""
    return _get_bin_left_edges(h.GetXaxis()), _get_buffer_hist1(h, flows=False)


def _get_buffers_graph(g: "ROOT.TGraph") -> Tuple[NDArray, NDArray]:
    """Get TGraph x and y buffers"""
    npoints = g.GetN()
    if npoints == 0:
        raise RuntimeError("Got graph with 0 points")

    return (
        frombuffer(g.GetX(), dtype=double, count=npoints).copy(),
        frombuffer(g.GetY(), dtype=double, count=npoints).copy(),
    )


def _get_buffers(obj):
    import ROOT

    if isinstance(obj, ROOT.TH1) and obj.GetDimension() == 1:
        return _get_buffers_hist1(obj)
    if isinstance(obj, ROOT.TGraph):
        return _get_buffers_graph(obj)

    raise RuntimeError(f"Do not know how to get buffers from {obj}")
