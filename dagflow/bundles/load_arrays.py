from typing import Optional, Mapping
from schema import Schema, Optional as SchemaOptional, And

from ..tools.schema import IsStrSeqOrStr, IsFilenameSeqOrFilename, AllFileswithExt
from ..storage import NodeStorage
from ..lib.Array import Array

_extensions = "root", "hdf5", "tsv", "txt"
_load_arrays_cfg = Schema({
    "name": str,
    "filenames": And(IsFilenameSeqOrFilename, AllFileswithExt(*_extensions)),
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

    for ext in _extensions:
        if filenames[0].endswith(f".{ext}"):
            break

    loader = _loaders.get(ext)
    match_filename = len(filenames)>1

    # storage = NodeStorage(default_containers=True)
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

        # with storage:
        mesh, _ = Array.make_stored(f"{name}.x.{skey}", x)
        Array.make_stored(f"{name}.y.{skey}", y, meshes=mesh)

    # NodeStorage.update_current(storage, strict=True)

def _load_tsv(filename: str, name: str):
    from numpy import loadtxt
    return loadtxt(filename, unpack=True)

def _load_hdf5(filename: str, name: str):
    from h5py import File
    file = File(filename, 'r')
    try:
        data = file[name]
    except KeyError as e:
        raise RuntimeError(f"Unable to read {name} from {filename}") from e

    cols = data.dtype.names
    return data[cols[0]], data[cols[1]]

def _load_root(filename: str, name: str):
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
