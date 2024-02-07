from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from multikeydict.typing import TupleKey

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


def _load_hist_data(
    acfg: Mapping | None = None, **kwargs
) -> tuple[tuple, tuple, NDArray | None, dict[TupleKey, tuple[NDArray, NDArray]]]:
    acfg = dict(acfg or {}, **kwargs)
    cfg = _validate_cfg(acfg)

    name = cfg["name"]
    filenames = cfg["filenames"]
    keys = cfg["replicate"]
    file_keys = cfg["replicate_files"]
    objectname = cfg["objects"]
    skip = cfg["skip"]
    index_order = cfg["index_order"]
    normalize = cfg["normalize"]

    xname = name, cfg["x"]
    yname = name, cfg["y"]

    edges_list: list[NDArray] = []
    data = {}
    for _, filename, _, key in iterate_filenames_and_objectnames(
        filenames, file_keys, keys, skip=skip, index_order=index_order
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

        return xname, yname, x0, data

    return xname, yname, None, data


def load_hist(acfg: Mapping | None = None, **kwargs):
    xname, yname, edges_common, data = _load_hist_data(acfg, **kwargs)

    storage = NodeStorage(default_containers=True)
    with storage:
        if edges_common is not None:
            edges, _ = Array.make_stored(strkey(xname), edges_common)
        else:
            edges = None

        for key, (x, y) in data.items():
            if edges_common is None:
                xkey = strkey(xname + key)
                edges, _ = Array.make_stored(xkey, x)
            ykey = strkey(yname + key)
            Array.make_stored(ykey, y, edges=edges)

    NodeStorage.update_current(storage, strict=True)

    return storage


def load_hist_data(acfg: Mapping | None = None, **kwargs):
    xname, yname, edges_common, data = _load_hist_data(acfg, **kwargs)

    storage = NodeStorage(default_containers=True)
    data_storage = storage("data")
    if edges_common is None:
        for key, (x, y) in data.items():
            data_storage[xname + key] = x
    else:
        data_storage[xname] = edges_common

    for key, (x, y) in data.items():
        data_storage[yname + key] = y

    NodeStorage.update_current(storage, strict=True)

    return storage
