from collections.abc import Callable, Mapping
from pathlib import Path

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
        SchemaOptional("name", default=()): And(str, Use(lambda s: (s,))),
        "filenames": And(IsFilenameSeqOrFilename, AllFileswithExt(*file_readers.keys())),
        "columns": Or([str], (str,), And(str, lambda s: (s,))),
        SchemaOptional("replicate", default=((),)): Or((IsStrSeqOrStr,), [IsStrSeqOrStr]),
        SchemaOptional("replicate_files", default=((),)): Or((IsStrSeqOrStr,), [IsStrSeqOrStr]),
        SchemaOptional("skip", default=None): And(
            Or(((str,),), [[str]]), Use(lambda l: tuple(set(k) for k in l))
        ),
        SchemaOptional("index_order", default=None): Or((int,), [int]),
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


def load_record(acfg: Mapping | None = None, **kwargs):
    acfg = dict(acfg or {}, **kwargs)
    cfg = _validate_cfg(acfg)

    name = cfg["name"]
    filenames = cfg["filenames"]
    keys = cfg["replicate"]
    file_keys = cfg["replicate_files"]
    objectname = cfg["objects"]
    skip = cfg["skip"]
    index_order = cfg["index_order"]
    columns = cfg["columns"]

    data = {}
    for _, filename, _, key in iterate_filenames_and_objectnames(
        filenames, file_keys, keys, skip=skip, index_order=index_order
    ):
        skey = strkey(key)
        logger.log(INFO3, f"Process {skey}")

        record = FileReader.record[filename, objectname(skey, key)]
        for column in columns:
            fullkey = (column,)+key
            data[fullkey] = record[column]

    storage = NodeStorage(default_containers=True)
    with storage:
        for key, record in data.items():
            Array.make_stored(name+key, record)

    NodeStorage.update_current(storage, strict=True)

    return storage
