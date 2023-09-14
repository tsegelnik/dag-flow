from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

from numpy import allclose
from numpy.typing import NDArray
from schema import And
from schema import Optional as SchemaOptional
from schema import Or, Schema, Use

from multikeydict.typing import TupleKey

from ..storage import NodeStorage
from ..tools.schema import (
    AllFileswithExt,
    IsReadable,
    IsFilenameSeqOrFilename,
    IsStrSeqOrStr,
    LoadFileWithExt,
    LoadYaml,
)

_extensions = {"root", "hdf5", "tsv", "txt"}
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


def load_graph_data(acfg: Optional[Mapping] = None, **kwargs):
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

    loader = _loaders.get(ext)
    if loader is None:
        raise RuntimeError(f"Unable to find loader for: {filenames[0]}")
    single_key = len(keys) == 1

    meshes = []
    data = {}
    for key in keys:
        filename = get_filename(filenames, key, single_key=single_key)

        skey = ".".join(key)
        iname = objects.get(skey, skey)
        x, y = loader(filename, iname)
        data[key] = x, y
        meshes.append(x)

    storage = NodeStorage(default_containers=True)
    data_storage = storage('data')

    if cfg["merge_x"]:
        x0 = meshes[0]
        for xi in meshes[1:]:
            if not allclose(x0, xi, atol=0, rtol=0):
                raise RuntimeError("load_graph: inconsistent x axes, unable to merge.")

        commonmesh = x0
        data_storage[xname] = commonmesh
    else:
        commonmesh = None

    with storage:
        for key, (x, y) in data.items():
            if commonmesh is None:
                data_storage[xname+key] = x

            data_storage[yname+key] = y

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


def _load_root(filename: str, name: str) -> Tuple[NDArray, NDArray]:
    from uproot import open

    file = open(filename)
    try:
        data = file[name]
    except KeyError as e:
        raise RuntimeError(f"Unable to read {name} from {filename}") from e

    y, x = data.to_numpy()

    return x[:-1], y


_loaders = {
    "txt": _load_tsv,
    "tsv": _load_tsv,
    "root": _load_root,
    "hdf5": _load_hdf5,
}
