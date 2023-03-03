from dictwrapper.dictwrapper import DictWrapper
from storage.storage import Storage

from typing import Any, Union
from schema import Schema, Or, Optional, Use, And, Schema, SchemaError

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


class ParsCfgHasProperFormat(object):
    def validate(self, data: dict) -> dict:
        format = data['format']
        if isinstance(format, str):
            nelements = 1
        else:
            nelements = len(format)

        dtin = DictWrapper(data)
        for key, subdata in dtin['variables'].walkitems():
            if isinstance(subdata, tuple):
                if len(subdata)==nelements: continue
            else:
                if nelements==1: continue

            key = ".".join(str(k) for k in key)
            raise SchemaError(f'Key "{key}" has  value "{subdata}"" inconsistent with format "{format}"')

        return data

IsNumber = Or(float, int, error='Invalid number "{}", expect int of float')
IsNumberOrTuple = Or(IsNumber, (IsNumber,), error='Invalid number/tuple {}')
IsLabel = Or(
    {'text': str, Optional('latex'): str},
    And(str, Use(lambda s: {'text': s}), error='Invalid string: {}')
)
IsValuesDict = NestedSchema(IsNumberOrTuple)
IsLabelsDict = NestedSchema(IsLabel, processdicts=True)
def IsFormatOk(format):
    if isinstance(format, tuple):
        if len(format)==1:
            f1,=format
            return f1=='value'
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

            return f1 in ('value', 'central')
    else:
        return format=='value'

IsFormat = Schema(IsFormatOk, error='Invalid variable format "{}".')
IsVarsCfgDict = Schema({
    'variables': IsValuesDict,
    'labels': IsLabelsDict,
    'format': IsFormat
    })

IsProperVarsCfg = And(IsVarsCfgDict, ParsCfgHasProperFormat())

def process_var_fixed1(vcfg, _, __):
    return {'central': vcfg, 'value': vcfg, 'sigma': None}

def process_var_fixed2(vcfg, format, hascentral) -> dict:
    ret = dict(zip(format, vcfg))
    if hascentral:
        ret.setdefault('value', ret['central'])
    else:
        ret.setdefault('central', ret['value'])
    ret['sigma'] = None
    return ret

def process_var_absolute(vcfg, format, hascentral) -> dict:
    ret = process_var_fixed2(vcfg, format, hascentral)
    ret['sigma'] = ret['sigma_absolute']
    return ret

def process_var_relative(vcfg, format, hascentral) -> dict:
    ret = process_var_fixed2(vcfg, format, hascentral)
    ret['sigma'] = ret['sigma_relative']*ret['central']
    return ret

def process_var_percent(vcfg, format, hascentral) -> dict:
    ret = process_var_fixed2(vcfg, format, hascentral)
    ret['sigma'] = 0.01*ret['sigma_percent']*ret['central']
    return ret

def get_format_processor(format):
    if isinstance(format, str):
        return process_var_fixed1

    errfmt = format[-1]
    if errfmt.startswith('sigma'):
        if errfmt.endswith('_absolute'):
            return process_var_absolute
        elif errfmt.endswith('_relative'):
            return process_var_relative
        else:
            return process_var_percent
    else:
        return process_var_fixed2

def iterate_varcfgs(cfg: DictWrapper):
    variablescfg = cfg['variables']
    labelscfg = cfg['labels']
    format = cfg['format']

    hascentral = 'central' in format
    process = get_format_processor(format)

    for key, varcfg in variablescfg.walkitems():
        varcfg = process(varcfg, format, hascentral)
        varcfg['label'] = labelscfg.get(key, None)
        yield key, varcfg

def load_variables(acfg):
    cfg = IsProperVarsCfg.validate(acfg)
    cfg = DictWrapper(cfg)

    ret = DictWrapper({})
    print('go')
    for key, varcfg in iterate_varcfgs(cfg):
        skey = '.'.join(key)
        print(skey, varcfg)


    return ret


cfg1 = {
        'variables': {
            'var1': 1.0,
            'var2': 1.0,
            'sub1': {
                'var3': 2.0
                }
            },
        'format': 'value',
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
            'var2': (1.0, 2.0, 0.1),
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
            'var2': (1.0, 2.0, 0.1),
            'sub1': {
                'var3': (2.0, 3.0, 0.1)
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
            'var1': (1.0, 1.0, 10),
            'var2': (1.0, 2.0, 10),
            'sub1': {
                'var3': (2.0, 3.0, 10)
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
            'var1': (1.0, 10),
            'var2': (2.0, 10),
            'sub1': {
                'var3': (3.0, 10)
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

# def test_load_variables_schema():
#     for cfg in cfgs:
#         cfg_schema.validate(cfg)

def test_load_variables():
    for cfg in cfgs:
        load_variables(cfg)
