from typing import Optional, Mapping, Tuple
from numpy.typing import NDArray
from schema import Schema, Optional as SchemaOptional, And

from ..tools.schema import IsStrSeqOrStr, IsFilenameSeqOrFilename, AllFileswithExt
from ..storage import NodeStorage
from ..lib.Array import Array

from numpy import allclose

_extensions = "root", "hdf5", "tsv", "txt"
_load_arrays_cfg = Schema({
    "name": str,
    "filenames": And(IsFilenameSeqOrFilename, AllFileswithExt(*_extensions)),
    SchemaOptional("merge_x", default=False): bool,
    SchemaOptional("x", default="x"): str,
    SchemaOptional("y", default="y"): str,
    SchemaOptional("replicate", default=((),)): (IsStrSeqOrStr,),
    SchemaOptional("objects", default={}): {str: str}
    })

def load_arrays(acfg: Optional[Mapping]=None, **kwargs):
    acfg = dict(acfg or {}, **kwargs)
    cfg = _load_arrays_cfg.validate(acfg)

    name = cfg["name"]
    filenames = cfg["filenames"]
    keys = cfg["replicate"]
    objects = cfg["objects"]

    xname = cfg['x']
    yname = cfg['y']

    for ext in _extensions:
        if filenames[0].endswith(f".{ext}"):
            break

    loader = _loaders.get(ext)
    match_filename = len(filenames)>1

    meshes = []
    data = {}
    for key in keys:
        if match_filename:
            for filename in filenames:
                if all(subkey in filename for subkey in key): break
            else:
                raise RuntimeError(f"Unable to find a file for {key}")
        else:
            filename = filenames[0]

        skey = '.'.join(key)
        iname = objects.get(skey, skey)
        x, y = loader(filename, iname)
        data[skey] = x, y
        meshes.append(x)

    if cfg["merge_x"]:
        x0 = meshes[0]
        for xi in meshes[1:]:
            if not allclose(x0, xi, atol=0, rtol=0):
                raise RuntimeError('load_arrays: inconsistent x axes, unable to merge.')

        commonmesh, _ = Array.make_stored(f"{name}.{xname}", x0)
    else:
        commonmesh = None

    storage = NodeStorage(default_containers=True)
    with storage:
        for key in keys:
            skey = '.'.join(key)
            x, y = data[skey]
            if commonmesh:
                mesh = commonmesh
            else:
                mesh, _ = Array.make_stored(f"{name}.{xname}.{skey}", x)
            Array.make_stored(f"{name}.{yname}.{skey}", y, meshes=mesh)

    NodeStorage.update_current(storage, strict=True)

    return storage

def _load_tsv(filename: str, name: str) -> Tuple[NDArray, NDArray]:
    from numpy import loadtxt
    return loadtxt(filename, unpack=True)

def _load_hdf5(filename: str, name: str) -> Tuple[NDArray, NDArray]:
    from h5py import File
    file = File(filename, 'r')
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
        'txt': _load_tsv,
        'tsv': _load_tsv,
        'root': _load_root,
        'hdf5': _load_hdf5,
        }
