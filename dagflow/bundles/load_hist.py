from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Optional

from numpy import allclose
from schema import And
from schema import Optional as SchemaOptional
from schema import Or, Schema, Use

from multikeydict.typing import strkey

from ..lib.Array import Array
from ..logger import INFO3, logger
from ..storage import NodeStorage
from ..tools.schema import (
    AllFileswithExt,
    IsFilenameSeqOrFilename,
    IsStrSeqOrStr,
    LoadFileWithExt,
    LoadYaml,
)
from .file_reader import FileReader, file_readers, iterate_filenames_and_objectnames

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
        SchemaOptional("skip", default=None): And(
            Or(((str,),), [[str]]), Use(lambda l: tuple(set(k) for k in l))
        ),
        SchemaOptional("objects", default=lambda: lambda st, tpl: st): Or(
            Callable, And({str: str}, Use(lambda dct: lambda st, tpl: dct.get(st, st)))
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


def load_hist(acfg: Optional[Mapping] = None, **kwargs):
    acfg = dict(acfg or {}, **kwargs)
    cfg = _validate_cfg(acfg)

    name = cfg["name"]
    filenames = cfg["filenames"]
    keys = cfg["replicate"]
    file_keys = cfg["replicate_files"]
    objectname = cfg["objects"]
    skip = cfg["skip"]
    normalize = cfg["normalize"]

    xname = name, cfg["x"]
    yname = name, cfg["y"]

    edges_list = []
    data = {}
    for _, filename, _, key in iterate_filenames_and_objectnames(
        filenames, file_keys, keys, skip=skip
    ):
        skey = strkey(key)
        logger.log(INFO3, f"Process {skey}")

        x, y = FileReader.hist[filename, objectname(skey, key)]
        if normalize and (ysum := y.sum()) != 0.0:
            y /= ysum
            logger.log(INFO3, "[normalize]")

        data[key] = x, y
        edges_list.append(x)

    if cfg["merge_x"]:
        x0 = edges_list[0]
        for xi in edges_list[1:]:
            if not allclose(x0, xi, atol=0, rtol=0):
                raise RuntimeError("load_hist: inconsistent x axes, unable to merge.")

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
