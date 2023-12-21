from contextlib import suppress
from os import R_OK, access
from pathlib import Path
from typing import Any, Callable, Sequence, Union

from schema import And, Or, Schema, SchemaError, Use

from ..logger import SUBINFO, logger

IsStrSeq = Or((str,), And([str], Use(tuple)))
IsStrSeqOrStr = Or(IsStrSeq, And(str, Use(lambda s: (s,))))


def IsReadable(filename: str):
    """Returns True if the file is readable"""
    return access(filename, R_OK)


IsFilename = Or(str, And(Path, Use(str)))

IsReadableFilename = And(IsFilename, IsReadable)

IsFilenameSeqOrFilename = Or(
    [IsFilename], (IsFilename,), And(IsFilename, Use(lambda s: (s,)))
)

IsReadableFilenameSeqOrFilename = Or(
    [IsReadableFilename],
    (IsReadableFilename,),
    And(IsReadableFilename, Use(lambda s: (s,))),
)


def IsFilewithExt(*exts: str):
    """Returns a function that retunts True if the file extension is consistent"""

    def checkfilename(filename: str):
        return filename.endswith(f".{ext}" for ext in exts)

    return checkfilename


def AllFileswithExt(*exts: str):
    """Returns a function that retunts True if the file extensions are consistent"""

    def checkfilenames(filenames: Sequence[str]):
        if not filenames:
            return False
        if isinstance(filenames, str):
            filename = filenames
            filenames = (filenames,)
        else:
            filename = filenames[0]
        for ext in exts:
            if filename.endswith(f".{ext}"):
                break
        else:
            return False
        return all(filename.endswith(f".{ext}") for filename in filenames)

    return checkfilenames


def LoadFileWithExt(
    *, key: Union[str, dict, None] = None, update: bool = False, **kwargs: Callable
):
    """Returns a function that retunts True if the file extension is consistent"""

    def checkfilename(filename: Union[str, dict]):
        if key is not None:
            dct = filename.copy()
            filename = dct.pop(key)
        else:
            dct = None
        for ext, loader in kwargs.items():
            if filename.endswith(f".{ext}"):
                break
        else:
            raise SchemaError(f"Do not know how to load {filename}")

        ret = loader(filename)
        if update and dct is not None:
            ret.update(dct)

        return ret

    return checkfilename


from yaml import Loader, load
from pathlib import Path


def LoadYaml(fname: Union[Path,str]):
    fname = str(fname)
    with open(fname, "r") as file:
        ret = load(file, Loader)

    logger.log(SUBINFO, f"Read: {fname}")
    return ret

import runpy


def LoadPy(fname: str, variable: str):
    logger.log(SUBINFO, f"Read: {fname} ({variable})")
    dct = runpy.run_path(fname)

    try:
        return dct[variable]
    except KeyError:
        raise RuntimeError(f"Variable {variable} is not provided in file {fname}")


def MakeLoaderPy(variable: str):
    def loader(fname):
        return LoadPy(fname, variable)

    return loader


from multikeydict.nestedmkdict import NestedMKDict


class NestedSchema:
    __slots__ = ("_schema", "_processdicts")
    _schema: Union[Schema, object]
    _processdicts: bool

    def __init__(self, /, schema: Union[Schema, object], *, processdicts: bool = False):
        self._schema = schema
        self._processdicts = processdicts

    def validate(self, data: Any) -> Any:
        if not isinstance(data, dict):
            return self._schema.validate(data)

        if self._processdicts:
            return {
                key: self._process_dict((key,), subdata)
                for key, subdata in data.items()
            }

        dtin = NestedMKDict(data)
        dtout = NestedMKDict({})
        for key, subdata in dtin.walkitems():
            dtout[key] = self._process_element(key, subdata)

        return dtout.object

    def _process_element(self, key, subdata: Any) -> Any:
        try:
            return self._schema.validate(subdata, _is_event_schema=False)
        except SchemaError as err:
            key = ".".join(str(k) for k in key)
            raise SchemaError(
                f'Key "{key}" has invalid value "{subdata}":\n{err.args[0]}'
            ) from err

    def _process_dict(self, key, data: Any) -> Any:
        if not isinstance(data, dict):
            return self._schema.validate(data)

        with suppress(SchemaError):
            return self._schema.validate(data, _is_event_schema=False)

        return {
            subkey: self._process_dict(key + (subkey,), subdata)
            for subkey, subdata in data.items()
        }
