from multikeydict.nestedmkdict import NestedMKDict
# from multikeydict.flatmkdict import FlatMKDict # To be used later
from gindex import GNIndex

from schema import Schema, Or, Optional, Use, And, Schema, SchemaError
from pathlib import Path

from ..tools.schema import NestedSchema, LoadFileWithExt, LoadYaml, MakeLoaderPy

class ParsCfgHasProperFormat(object):
    def validate(self, data: dict) -> dict:
        format = data['format']
        if isinstance(format, str):
            nelements = 1
        else:
            nelements = len(format)

        dtin = NestedMKDict(data)
        for key, subdata in dtin['parameters'].walkitems():
            if isinstance(subdata, tuple):
                if len(subdata)==nelements: continue
            else:
                if nelements==1: continue

            key = ".".join(str(k) for k in key)
            raise SchemaError(f'Key "{key}" has  value "{subdata}"" inconsistent with format "{format}"')

        return data

IsNumber = Or(float, int, error='Invalid number "{}", expect int of float')
IsNumberOrTuple = Or(IsNumber, (IsNumber,), And([IsNumber], Use(tuple)), error='Invalid number/tuple {}')
IsLabel = Or({
        'text': str,
        Optional('latex'): str,
        Optional('graph'): str,
        Optional('mark'): str,
        Optional('name'): str
    },
    And(str, Use(lambda s: {'text': s}), error='Invalid string: {}')
)
IsValuesDict = NestedSchema(IsNumberOrTuple)
IsLabelsDict = NestedSchema(IsLabel, processdicts=True)
def IsFormatOk(format):
    if not isinstance(format, (tuple, list)):
        return format=='value'

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

def CheckCorrelationSizes(cfg):
    nnames = len(cfg['names'])
    matrix = cfg['matrix']
    nrows = len(matrix)

    if nrows!=nnames:
        return False

    for row in matrix:
        if nnames!=len(row):
            return False

    return True

IsCorrelationsDict = And({
        'names': Or((str,), And([str], Use(tuple))),
        'matrix_type': Or('correlation', 'covariance'),
        'matrix': [[IsNumber]],
        }, CheckCorrelationSizes)
IsNestedCorrelationsDict = NestedSchema(IsCorrelationsDict, processdicts=True)

IsFormat = Schema(IsFormatOk, error='Invalid parameter format "{}".')
IsStrSeq = (str,)
IsStrSeqOrStr = Or(IsStrSeq, And(str, Use(lambda s: (s,))))
IsParsCfgDict = Schema({
    'parameters': IsValuesDict,
    'labels': IsLabelsDict,
    'format': IsFormat,
    'state': Or('fixed', 'variable', error='Invalid parameters state: {}'),
    Optional('path', default=''): str,
    Optional('replicate', default=((),)): (IsStrSeqOrStr,),
    Optional('replica_key_offset', default=0): int,
    Optional('correlations', default={}): IsNestedCorrelationsDict
    },
    # error = 'Invalid parameters configuration: {}'
)
IsProperParsCfgDict = And(IsParsCfgDict, ParsCfgHasProperFormat())
IsLoadableDict = And(
            {
                'load': Or(str, And(Path, Use(str))),
                Optional(str): object
            },
            Use(LoadFileWithExt(
                yaml=LoadYaml,
                py=MakeLoaderPy('configuration'),
                key='load',
                update=True
            ), error='Failed to load {}'),
            IsProperParsCfgDict
        )
def ValidateParsCfg(cfg):
    if isinstance(cfg, dict) and 'load' in cfg:
        return IsLoadableDict.validate(cfg)
    else:
        return IsProperParsCfgDict.validate(cfg)

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
    if not errfmt.startswith('sigma'):
        return process_var_fixed2

    if errfmt.endswith('_absolute'):
        return process_var_absolute
    elif errfmt.endswith('_relative'):
        return process_var_relative
    else:
        return process_var_percent

def format_latex(k, s: str, /, *args, **kwargs) -> str:
    if k=='latex' and '$' in s:
        return s

    return s.format(*args, **kwargs)

def format_dict(dct: dict, /, *args, **kwargs) -> dict:
    return {
        k: format_latex(k, v, *args, **kwargs) for k, v in dct.items()
    }

def get_label(key: tuple, labelscfg: dict) -> dict:
    try:
        return labelscfg[key]
    except KeyError:
        pass

    for n in range(1, len(key)+1):
        subkey = key[:-n]
        try:
            lcfg = labelscfg[subkey]
        except KeyError:
            continue

        if not subkey and not 'text' in lcfg:
            break

        sidx = '.'.join(key[n-1:])
        return format_dict(lcfg, sidx)

    return {}

def iterate_varcfgs(cfg: NestedMKDict):
    parameterscfg = cfg['parameters']
    labelscfg = cfg['labels']
    format = cfg['format']

    hascentral = 'central' in format
    process = get_format_processor(format)

    for key, varcfg in parameterscfg.walkitems():
        varcfg = process(varcfg, format, hascentral)
        varcfg['label'] = get_label(key, labelscfg)
        yield key, varcfg

from dagflow.parameters import Parameters
from dagflow.lib.SumSq import SumSq

from numpy.typing import ArrayLike
from numpy import ascontiguousarray
from typing import Sequence
class CorrelationsDef:
    __slots__ = ('matrix_type', 'matrix', 'names')

    def __init__(
        self,
        matrix_type: str,
        matrix: ArrayLike,
        names: Sequence[str]
    ):
        self.matrix_type = matrix_type
        self.matrix = ascontiguousarray(matrix, dtype='d')
        self.names = names

def load_parameters(acfg):
    cfg = ValidateParsCfg(acfg)
    cfg = NestedMKDict(cfg)

    pathstr = cfg['path']
    if pathstr:
        path = tuple(pathstr.split('.'))
    else:
        path = ()

    state = cfg['state']

    ret = NestedMKDict(
        {
            'parameter': {
                'constant': {},
                'free': {},
                'constrained': {},
                'normalized': {},
                },
            'stat': {
                'nuisance_parts': {},
                'nuisance': {},
                },
            'parameter_node': {
                'constant': {},
                'free': {},
                'constrained': {}
                }
        },
        sep='.'
    )

    subkeys = cfg['replicate']
    replica_key_offset = cfg['replica_key_offset']
    if replica_key_offset>0:
        make_key = lambda key, subkey: key[:-replica_key_offset]+subkey+key[replica_key_offset:]
    elif replica_key_offset==0:
        make_key = lambda key, subkey: key+subkey
    else:
        raise ValueError('{replica_key_offset=} should be non-negative')

    varcfgs = NestedMKDict({})
    normpars = {}
    for key_general, varcfg in iterate_varcfgs(cfg):
        varcfg.setdefault(state, True)

        label_general = varcfg['label']

        for subkey in subkeys:
            key = key_general + subkey
            key = make_key(key_general, subkey)
            key_str = '.'.join(key)
            subkey_str = '.'.join(subkey)

            label = format_dict(
                label_general.copy(),
                subkey=subkey_str,
                space_subkey=f' {subkey_str}',
                subkey_space=f'{subkey_str} ',
            )
            varcfg['label'] = label
            label['key'] = key_str
            label.setdefault('text', key_str)

            varcfgs[key] = (varcfg,) # protect dictionary from being 'nested'

    pars = NestedMKDict({})
    for key, (varcfg,) in varcfgs.walkitems():
        par = Parameters.from_numbers(**varcfg)
        pars[key] = par

    for key, par in pars.walkitems():
        if par.is_constrained:
            target = ('constrained', path)
        elif par.is_fixed:
            target = ('constant', path)
        else:
            target = ('free', path)

        targetkey = target+key
        ret[('parameter_node',)+targetkey] = par

        ptarget = ('parameter', targetkey)
        for subname, subpar in par.iter_items():
            ret[ptarget+subname] = subpar

        normpars_i = normpars.setdefault(key[0], [])
        ntarget = ('parameter', 'normalized', path)+key
        for subname, subpar in par.iter_norm_items():
            ret[ntarget+subname] = subpar

            normpars_i.append(subpar)

        for name, np in normpars.items():
            if not np:
                continue

            ssq = SumSq(f'nuisance for {pathstr}.{name}')
            (n.output for n in np) >> ssq
            ssq.close()
            ret[('stat', 'nuisance_parts', path, name)] = ssq

    return ret
