from dictwrapper.dictwrapper import DictWrapper
from storage.storage import Storage

from typing import Any, Union
from schema import Schema, Or, Optional, Use, And, Schema, SchemaError

class NestedSchema(object):
    __slots__ = ('_schema',)
    _schema: Union[Schema,object]

    def __init__(self, schema: Union[Schema,object]):
        self._schema = schema

    def validate(self, data: Any) -> Any:
        if not isinstance(data, dict):
            return self._schema.validate(data)

        dtin = DictWrapper(data)
        dtout = DictWrapper({})
        for key, subdata in dtin.walkitems():
            try:
                subdata_new = self._schema.validate(subdata, _is_event_schema=False)
            except SchemaError as err:
                key = ".".join(str(k) for k in key)
                raise SchemaError(f'The key "{key}" has invalid value "{subdata}":\n{err.args[0]}') from err
            dtout[key] = subdata_new

        return dtout.object

class ParsCfgHasProperFormat(object):
    def validate(self, data: dict) -> dict:
        format = data['format']
        nelements = len(format)

        dtin = DictWrapper(data)
        for key, subdata in dtin['variables'].walkitems():
            if isinstance(subdata, tuple):
                if len(subdata)==nelements: continue
            else:
                if nelements==1: continue

            key = ".".join(str(k) for k in key)
            raise SchemaError(f'The key "{key}" has  value "{subdata}"" inconsistent with format "{format}"')

        return data

IsNumber = Or(float, int, error='Invalid number "{}", expect int of float')
IsNumberOrTuple = Or(IsNumber, (IsNumber,), error='Invalid number/tuple {}')
IsLabel = Or(
    {'text': str, Optional('latex'): str},
    And(str, Use(lambda s: {'text': s}), error='Invalid string')
)
IsValuesDict = NestedSchema(IsNumberOrTuple)
IsLabelsDict = NestedSchema(IsLabel)
def IsFormatOk(format):
    f1, f2, f3 = None, None, None

    if isinstance(format, tuple):
        if len(format)==1:
            f1,=format
        else:
            if len(format)==2:
                f1, f3 = format
            elif len(format)==3:
                f1, f2, f3 = format

                if f2 not in ('value', 'central') or f1==f2:
                    return False
            else:
                return False

            if f3 not in ('sigma_absolute', 'sigma_relative', 'sigma_percent'):
                return False
    else:
        f1 = format

    if f1 not in ('value', 'central'):
        return True

    return True
IsFormat = Schema(IsFormatOk, error='Invalid variable format "{}".')

cfg_schema = Schema(
        And({
                'variables': IsValuesDict,
                'labels': IsLabelsDict,
                'format': IsFormat
            },
            ParsCfgHasProperFormat()
            )
        )

cfg1 = {
        'variables': {
            'var1': 1.0,
            'var2': 1.0,
            'sub1': {
                'var3': 2.0
                }
            },
        'format': ('value',),
        'labels': {
            'var1': {
                'text': 'text label 1',
                'latex': r'\LaTeX label 1',
                },
            'var2': 'simple label 2',
            },
        }

cfg2 = {
        'variables': {
            'var1': (1.0, 1.0, 0.1),
            'var2': (1.0, 1.0, 0.1),
            'sub1': {
                'var3': (2.0, 1.0, 0.1)
                }
            },
        'format': ('value', 'central', 'sigma_absolute'),
        'labels': {
            'var1': {
                'text': 'text label 1',
                'latex': r'\LaTeX label 1',
                },
            'var2': 'simple label 2'
            },
        }

cfg3 = {
        'variables': {
            'var1': (1.0, 1.0, 0.1),
            'var2': (1.0, 1.0, 0.1),
            'sub1': {
                'var3': (2.0, 1.0, 0.1)
                }
            },
        'labels': {
            'var1': {
                'text': 'text label 1',
                'latex': r'\LaTeX label 1',
                },
            'var2': 'simple label 2'
            },
        'format': ('value', 'central', 'sigma_relative')
        }

cfg4 = {
        'variables': {
            'var1': (1.0, 1.0, 0.1),
            'var2': (1.0, 1.0, 0.1),
            'sub1': {
                'var3': (2.0, 1.0, 0.1)
                }
            },
        'labels': {
            'var1': {
                'text': 'text label 1',
                'latex': r'\LaTeX label 1',
                },
            'var2': 'simple label 2'
            },
        'format': ('value', 'central', 'sigma_percent')
        }

cfg5 = {
        'variables': {
            'var1': (1.0, 0.1),
            'var2': (1.0, 0.1),
            'sub1': {
                'var3': (1.0, 0.1)
                }
            },
        'labels': {
            'var1': {
                'text': 'text label 1',
                'latex': r'\LaTeX label 1',
                },
            'var2': 'simple label 2'
            },
        'format': ('central', 'sigma_percent')
        }

cfgs = (cfg1, cfg2, cfg3, cfg4, cfg5)

def test_load_variables_schema():
    for cfg in cfgs:
        cfg_schema.validate(cfg)

def test_load_variables():
    pass
