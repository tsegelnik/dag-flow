from ..logger import logger, SUBINFO
from typing import Any, Union
from schema import Schema, Schema, SchemaError
from contextlib import suppress

from os import access, R_OK
from typing import Callable

def IsReadable(filename: str):
    """Returns True if the file is readable"""
    return access(filename, R_OK)

def IsFilewithExt(*exts: str):
    """Returns a function that retunts True if the file extension is consistent"""
    def checkfilename(filename: str):
        return filename.endswith(f'.{ext}' for ext in exts)
    return checkfilename

def LoadFileWithExt(*, key: Union[str, dict, None]=None, update: bool=False, **kwargs: Callable):
    """Returns a function that retunts True if the file extension is consistent"""
    def checkfilename(filename: Union[str, dict]):
        if key is not None:
            dct = filename.copy()
            filename = dct.pop(key)
        else:
            dct = None
        for ext, loader in kwargs.items():
            if filename.endswith(f'.{ext}'):
                ret = loader(filename)

                if update and dct is not None:
                    ret.update(dct)

                return ret

        raise SchemaError(f'Do not know how to load {filename}')
    return checkfilename

from yaml import load, Loader
def LoadYaml(fname: str):
    logger.log(SUBINFO, f'Read: {fname}')
    with open(fname, 'r') as file:
        return load(file, Loader)

import runpy
def LoadPy(fname: str, variable: str):
    logger.log(SUBINFO, f'Read: {fname} ({variable})')
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
class NestedSchema(object):
    __slots__ = ('_schema', '_processdicts')
    _schema: Union[Schema,object]
    _processdicts: bool

    def __init__(self, /, schema: Union[Schema,object], *, processdicts: bool=False):
        self._schema = schema
        self._processdicts = processdicts

    def validate(self, data: Any) -> Any:
        if not isinstance(data, dict):
            return self._schema.validate(data)

        if self._processdicts:
            return {
                key: self._process_dict((key,), subdata) for key, subdata in data.items()
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
            raise SchemaError(f'Key "{key}" has invalid value "{subdata}":\n{err.args[0]}') from err

    def _process_dict(self, key, data: Any) -> Any:
        if not isinstance(data, dict):
            return self._schema.validate(data)

        with suppress(SchemaError):
            return self._schema.validate(data, _is_event_schema=False)

        return {
            subkey: self._process_dict(key+(subkey,), subdata) for subkey, subdata in data.items()
        }
