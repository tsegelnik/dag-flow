from collections.abc import Callable, Mapping
from pathlib import Path

from numpy import asarray
from schema import And, Optional, Or, Schema, Use

from multikeydict.typing import strkey

from ..lib.common import Array
from ..storage import NodeStorage
from ..tools.logger import INFO3, logger
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
        Optional("dtype", default=None): Or("d", "f"),
        Optional("replicate_outputs", default=((),)): Or((IsStrSeqOrStr,), [IsStrSeqOrStr]),
        Optional("replicate_files", default=((),)): Or((IsStrSeqOrStr,), [IsStrSeqOrStr]),
        Optional("skip", default=None): And(
            Or(((str,),), [[str]]), Use(lambda l: tuple(set(k) for k in l))
        ),
        Optional("key_order", default=None): Or((int,), [int]),
        Optional("objects", default=lambda: lambda st, tpl: st): Or(
            Callable, And({str: str}, Use(lambda dct: lambda st, tpl: dct.get(st, st)))
        ),
    }
)

_schema_loadable_cfg = And(
    {"load": Or(str, And(Path, Use(str))), Optional(str): object},
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


def load_array(acfg: Mapping | None = None, *, array_kwargs: Mapping = {}, **kwargs) -> NodeStorage:
    acfg = dict(acfg or {}, **kwargs)
    cfg = _validate_cfg(acfg)

    name = (cfg["name"],)
    filenames = cfg["filenames"]
    keys = cfg["replicate_outputs"]
    file_keys = cfg["replicate_files"]
    objectname = cfg["objects"]
    skip = cfg["skip"]
    key_order = cfg["key_order"]
    dtype = cfg["dtype"]

    data = {}
    for _, filename, _, key in iterate_filenames_and_objectnames(
        filenames, file_keys, keys, skip=skip, key_order=key_order
    ):
        skey = strkey(key)
        logger.log(INFO3, f"Process {skey}")

        array = FileReader.array[filename, objectname(skey, key)]
        data[key] = asarray(array, dtype)

    storage = NodeStorage(default_containers=True)
    with storage:
        for key, array in data.items():
            Array.replicate(name=".".join(name + key), array=array, **array_kwargs)

    NodeStorage.update_current(storage, strict=True)

    return storage
