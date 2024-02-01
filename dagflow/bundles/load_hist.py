from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Optional

from numpy import allclose
from schema import And
from schema import Optional as SchemaOptional
from schema import Or, Schema, Use

from multikeydict.typing import KeyLike, properkey, strkey

from .file_reader import FileReader, file_readers
from ..lib.Array import Array
from ..storage import NodeStorage
from ..tools.schema import (
    AllFileswithExt,
    IsFilenameSeqOrFilename,
    IsStrSeqOrStr,
    LoadFileWithExt,
    LoadYaml,
)

_schema_cfg = Schema(
    {
        "name": str,
        "filenames": And(IsFilenameSeqOrFilename, AllFileswithExt(*file_readers.keys())),
        SchemaOptional("merge_x", default=False): bool,
        SchemaOptional("x", default="x"): str,
        SchemaOptional("y", default="y"): str,
        SchemaOptional("normalize", default=False): bool,
        SchemaOptional("replicate", default=((),)): Or((IsStrSeqOrStr,), [IsStrSeqOrStr]),
        SchemaOptional("replicate_files", default=((),)): Or((IsStrSeqOrStr,), [IsStrSeqOrStr]),
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

def iterate_files(filenames: Sequence[str | Path], keys: Sequence[KeyLike]):
    for keylike in keys:
        key = properkey(keylike)
        for afilename in filenames:
            filename = str(afilename)
            if '{' in filename:
                 ffilename = filename.format(*key)
                 yield key, ffilename
                 break
            elif all(map(filename.__contains__, key)):
                yield key, afilename
                break
        else:
            raise RuntimeError(f"Could not find a file for key {'.'.join(key)}")

def _validate_cfg(cfg):
    if isinstance(cfg, dict) and "load" in cfg:
        return _schema_loadable_cfg.validate(cfg)
    else:
        return _schema_cfg.validate(cfg)


def load_hist(acfg: Optional[Mapping] = None, **kwargs):
    acfg = dict(acfg or {}, **kwargs)
    cfg = _validate_cfg(acfg)

    name = cfg["name"]
    filenames = cfg["filenames"]
    keys = cfg["replicate"]
    file_keys = cfg["replicate_files"]
    objectname = cfg["objects"]
    normalize = cfg["normalize"]

    xname = name, cfg["x"]
    yname = name, cfg["y"]

    edges_list = []
    data = {}
    for filekey, filename in iterate_files(filenames, file_keys):
        sfilekey = strkey(filekey),
        for key in keys:
            skey = strkey(sfilekey+key)
            x, y = FileReader.hist[filename, objectname(skey)]
            if normalize and (ysum:=y.sum())!=0.0:
                y /= ysum

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
