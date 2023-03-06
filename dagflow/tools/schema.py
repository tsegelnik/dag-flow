from typing import Any, Union
from schema import Schema, Schema, SchemaError
from dictwrapper.dictwrapper import DictWrapper

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
            dtout = {}
            for key, subdata in data.items():
                dtout[key] = self._process_dict((key,), subdata)

            return dtout

        dtin = DictWrapper(data)
        dtout = DictWrapper({})
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

        try:
            return self._schema.validate(data, _is_event_schema=False)
        except SchemaError:
            pass

        return {
            subkey: self._process_dict(key+(subkey,), subdata) for subkey, subdata in data.items()
        }
