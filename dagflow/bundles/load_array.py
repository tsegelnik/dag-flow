from pathlib import Path
from typing import Mapping, Optional, Sequence

from numpy.typing import NDArray
from schema import And
from schema import Optional as SchemaOptional
from schema import Or, Schema, Use

from multikeydict.typing import TupleKey

from ..logger import SUBINFO, logger
from ..lib.Array import Array
from ..storage import NodeStorage
from ..tools.schema import (
    AllFileswithExt,
    IsReadable,
    IsFilenameSeqOrFilename,
    IsStrSeqOrStr,
    LoadFileWithExt,
    LoadYaml,
)

_extensions = {"root", "hdf5", "tsv", "txt", "npz"}
_schema_cfg = Schema(
    {
        "name": str,
        "filenames": And(IsFilenameSeqOrFilename, AllFileswithExt(*_extensions)),
        SchemaOptional("merge_x", default=False): bool,
        SchemaOptional("x", default="x"): str,
        SchemaOptional("y", default="y"): str,
        SchemaOptional("replicate", default=((),)): Or(
            (IsStrSeqOrStr,), [IsStrSeqOrStr]
        ),
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
    filenames: Sequence[str], key: TupleKey, *, single_key: bool = False
) -> str:
    if single_key and len(filenames) == 1:
        return filenames[0]
    checked_filenames = []
    for filename in filenames:
        if "{key}" in filename:
            ifilename = filename.format(key="_".join(key))
            checked_filenames.append(ifilename)
            if IsReadable(ifilename):
                return ifilename
        elif all(subkey in filename for subkey in key):
            checked_filenames.append(filename)
            if IsReadable(filename):
                return filename

    raise RuntimeError(f"Unable to find readable filename for {key}. Checked: {checked_filenames}")


def load_array(acfg: Optional[Mapping] = None, **kwargs):
    acfg = dict(acfg or {}, **kwargs)
    cfg = _validate_cfg(acfg)

    name = (cfg["name"],)
    filenames = cfg["filenames"]
    keys = cfg["replicate"]
    objects = cfg["objects"]

    try:
        ext = next(ext for ext in _extensions if filenames[0].endswith(f".{ext}"))
    except StopIteration:
        raise RuntimeError(f"Unable to find extension for: {filenames[0]}")

    loader = _loaders.get(ext)
    if loader is None:
        raise RuntimeError(f"Unable to find loader for: {filenames[0]}")
    single_key = len(keys) == 1

    data = {}
    for key in keys:
        filename = get_filename(filenames, key, single_key=single_key)

        skey = ".".join(key)
        iname = objects.get(skey, skey)
        data[name + key] = loader(filename, iname)
        logger.log(SUBINFO, f"Read: {filename}")
    storage = NodeStorage(default_containers=True)
    with storage:
        for key, array in data.items():
            Array.make_stored(key, array)

    NodeStorage.update_current(storage, strict=True)

    return storage


def _load_tsv(filename: str, _: str) -> NDArray:
    from numpy import loadtxt

    return loadtxt(filename)


def _load_hdf5(filename: str, name: str) -> NDArray:
    if not name:
        raise RuntimeError(f"Need an object name to read from {filename}")
    from h5py import File

    file = File(filename, "r")
    try:
        data = file[name]
    except KeyError as e:
        raise RuntimeError(f"Unable to read {name} from {filename}") from e

    return data[:]


def _load_npz(filename: str, name: str) -> NDArray:
    if not name:
        raise RuntimeError(f"Need an object name to read from {filename}")
    from numpy import load

    file = load(filename)
    try:
        data = file[name]
    except KeyError as e:
        raise RuntimeError(f"Unable to read {name} from {filename}") from e

    return data


def _load_root_uproot(filename: str, name: str) -> NDArray:
    if not name:
        raise RuntimeError(f"Need an object name to read from {filename}")
    from uproot import open

    file = open(filename)
    try:
        data = file[name]
    except KeyError as e:
        raise RuntimeError(f"Unable to read {name} from {filename}") from e

    return data.to_numpy()


def _load_root_ROOT(filename: str, name: str) -> NDArray:
    if not name:
        raise RuntimeError(f"Need an object name to read from {filename}")
    from ROOT import TFile

    file = TFile(filename)
    if file.IsZombie():
        raise RuntimeError(f"Can not open file {filename}")
    data = file.Get(name)
    if not data:
        raise RuntimeError(f"Unable to read {name} from {filename}") from e

    return _get_buffer(data)


def _load_root(filename: str, *args, **kwargs) -> NDArray:
    try:
        return _load_root_uproot(filename, *args, **kwargs)
    except AttributeError:
        pass

    try:
        return _load_root_ROOT(filename, *args, **kwargs)
    except ImportError:
        raise RuntimeError(
            f"Error reading file {filename}. `uproot` is unable, `ROOT` is not found."
        )


_loaders = {
    "txt": _load_tsv,
    "tsv": _load_tsv,
    "root": _load_root,
    "hdf5": _load_hdf5,
    "npz": _load_npz,
}

from numpy import dtype, frombuffer


def _get_buffer_hist1(h, flows=False):
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
    res = frombuffer(buf, dtype(buf.typecode), (nx + 2) * (ny + 2)).reshape(
        (ny + 2, nx + 2)
    )
    if not flows:
        res = res[1 : ny + 1, 1 : nx + 1]

    return res.copy()


def _get_buffer_matrix(m):
    """Get TMatrix buffer"""
    cbuf = m.GetMatrixArray()
    res = frombuffer(cbuf, dtype(cbuf.typecode), m.GetNoElements()).reshape(
        m.GetNrows(), m.GetNcols()
    )
    return res.copy()


def _get_buffer(obj):
    import ROOT

    if isinstance(obj, ROOT.TH1) and obj.GetDimension() == 1:
        return _get_buffer_hist1(obj)
    if isinstance(obj, ROOT.TH2) and obj.GetDimension() == 2:
        return _get_buffer_hist2(obj)
    if isinstance(obj, (ROOT.TMatrixD, ROOT.TMatrixF)):
        return _get_buffer_matrix(obj)

    raise Exception()
