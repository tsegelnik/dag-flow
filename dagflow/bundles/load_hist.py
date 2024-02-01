from collections.abc import Callable, Mapping, Sequence
from os.path import basename
from pathlib import Path
from typing import Optional

from numpy import allclose
from schema import And
from schema import Optional as SchemaOptional
from schema import Or, Schema, Use

from multikeydict.typing import TupleKey

from .file_reader import file_readers
from ..lib.Array import Array
from ..logger import SUBINFO, logger
from ..storage import NodeStorage
from ..tools.schema import (
    AllFileswithExt,
    IsFilenameSeqOrFilename,
    IsReadable,
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


def _validate_cfg(cfg):
    if isinstance(cfg, dict) and "load" in cfg:
        return _schema_loadable_cfg.validate(cfg)
    else:
        return _schema_cfg.validate(cfg)


def get_filename(
    filenames: Sequence[str],
    key: TupleKey,
    *,
    single_key: bool = False,
    multiple_files: bool = False,
) -> str:
    if (single_key or not multiple_files) and len(filenames) == 1:
        return filenames[0]
    checked_filenames = []
    skey = "_".join(key)
    for filename in filenames:
        if Path(filename).is_dir():
            if filename.endswith(".tsv"):
                ext = filename[-3:]
                bname = basename(filename[:-4])
                ifilename = f"{filename}/{bname}_{skey}.{ext}"
                checked_filenames.append(ifilename)
                if IsReadable(ifilename):
                    return ifilename
        elif "{key}" in filename:
            ifilename = filename.format(key=skey)
            checked_filenames.append(ifilename)
            if IsReadable(ifilename):
                return ifilename
        elif all(subkey in filename for subkey in key):
            checked_filenames.append(filename)
            if IsReadable(filename):
                return filename

    raise RuntimeError(f"Unable to find readable filename for {key}. Checked: {checked_filenames}")


def load_hist(acfg: Optional[Mapping] = None, **kwargs):
    acfg = dict(acfg or {}, **kwargs)
    cfg = _validate_cfg(acfg)

    name = cfg["name"]
    filenames = cfg["filenames"]
    keys = cfg["replicate"]
    objects = cfg["objects"]
    normalize = cfg["normalize"]

    xname = name, cfg["x"]
    yname = name, cfg["y"]

    try:
        ext = next(ext for ext in file_readers if filenames[0].endswith(f".{ext}"))
    except StopIteration:
        raise RuntimeError(f"Unable to find extension for: {filenames[0]}")

    loader, multiple_files = file_readers.get(ext)
    if loader is None:
        raise RuntimeError(f"Unable to find loader for: {filenames[0]}")
    single_key = len(keys) == 1

    edges_list = []
    data = {}
    for key in keys:
        filename = get_filename(
            filenames, key, single_key=single_key, multiple_files=multiple_files
        )

        skey = ".".join(key)
        iname = objects(skey)
        x, y = loader(filename, iname)
        logger.log(SUBINFO, f"Read: {filename}")

        if normalize:
            y /= y.sum()

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
