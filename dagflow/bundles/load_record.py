from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from numpy import asarray
from schema import And, Optional, Or, Schema, Use

from multikeydict.tools import reorder_key
from multikeydict.typing import strkey

from ..lib.base import Array
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

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from multikeydict.typing import TupleKey

_schema_cfg = Schema(
    {
        Optional("name", default=()): And(str, Use(lambda s: (s,))),
        "filenames": And(IsFilenameSeqOrFilename, AllFileswithExt(*file_readers.keys())),
        "columns": Or([str], (str,), And(str, lambda s: (s,))),
        Optional("dtype", default=None): Or("d", "f"),
        Optional("replicate_outputs", default=((),)): Or((IsStrSeqOrStr,), [IsStrSeqOrStr]),
        Optional("replicate_files", default=((),)): Or((IsStrSeqOrStr,), [IsStrSeqOrStr]),
        Optional("skip", default=None): And(
            Or((Or((str,), {str}),), [Or([str], {str})]), Use(lambda l: tuple(set(k) for k in l))
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


def _load_record_data(
    acfg: Mapping | None = None, **kwargs
) -> tuple[TupleKey, dict[TupleKey, NDArray]]:
    acfg = dict(acfg or {}, **kwargs)
    cfg = _validate_cfg(acfg)

    name = cfg["name"]
    filenames = cfg["filenames"]
    keys = cfg["replicate_outputs"]
    file_keys = cfg["replicate_files"]
    objectname = cfg["objects"]
    skip = cfg["skip"]
    key_order = cfg["key_order"]
    dtype = cfg["dtype"]
    columns = cfg["columns"]

    data: dict[TupleKey, NDArray] = {}
    for _, filename, _, key in iterate_filenames_and_objectnames(
        filenames, file_keys, keys, skip=skip
    ):
        skey = strkey(key)
        logger.log(INFO3, f"Process {skey}")

        record = FileReader.record[filename, objectname(skey, key)]
        for column in columns:
            fullkey = reorder_key((column,) + key, key_order)
            rec = record[column][:]
            data[fullkey] = asarray(rec, dtype)

    return name, data


def load_record(acfg: Mapping | None = None, **kwargs) -> NodeStorage:
    name, data = _load_record_data(acfg, **kwargs)

    storage = NodeStorage(default_containers=True)
    with storage:
        for key, record in data.items():
            Array.replicate(name=strkey(name + key), array=record)

    NodeStorage.update_current(storage, strict=True)

    return storage


def load_record_data(acfg: Mapping | None = None, **kwargs) -> NodeStorage:
    name, data = _load_record_data(acfg, **kwargs)

    storage = NodeStorage(default_containers=True)
    data_storage = storage("data")
    for key, record in data.items():
        data_storage[name + key] = record

    NodeStorage.update_current(storage, strict=True)

    return storage
