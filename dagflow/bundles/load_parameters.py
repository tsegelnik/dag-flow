from collections.abc import Generator
from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from typing import Optional as OptionalType

from multikeydict.nestedmkdict import NestedMKDict
from multikeydict.typing import properkey
from schema import And
from schema import Optional
from schema import Or
from schema import Schema
from schema import SchemaError
from schema import Use

from ..exception import InitializationError
from ..labels import format_latex
from ..labels import inherit_labels
from ..storage import NodeStorage
from ..tools.schema import IsStrSeqOrStr
from ..tools.schema import LoadFileWithExt
from ..tools.schema import LoadYaml
from ..tools.schema import MakeLoaderPy
from ..tools.schema import NestedSchema


class ParsCfgHasProperFormat:
    def validate(self, data: dict) -> dict:
        format = data["format"]
        nelements = 1 if isinstance(format, str) else len(format)

        dtin = NestedMKDict(data)
        for key, subdata in dtin("parameters").walkitems():
            if isinstance(subdata, tuple):
                if len(subdata) == nelements:
                    continue
            else:
                if nelements == 1:
                    continue

            key = ".".join(str(k) for k in key)
            raise SchemaError(
                f'Key "{key}" has  value "{subdata}"" inconsistent with format "{format}"'
            )

        return data


IsNumber = Or(float, int, error='Invalid number "{}", expect int of float')
IsNumberOrTuple = Or(
    IsNumber, (IsNumber,), And([IsNumber], Use(tuple)), error="Invalid number/tuple {}"
)
label_keys = {"text", "latex", "graph", "mark", "name", "index_values"}
IsLabel = Or(
    {
        "text": str,
        Optional("latex"): str,
        Optional("graph"): str,
        Optional("mark"): str,
        Optional("name"): str,
    },
    And(str, Use(lambda s: {"text": s}), error="Invalid string: {}"),
)
IsValuesDict = NestedSchema(IsNumberOrTuple)
IsLabelsDict = NestedSchema(IsLabel, processdicts=True)


def IsFormatOk(format):
    if not isinstance(format, (tuple, list)):
        return format == "value"

    match format:
        case "value" | ["value"]:
            return True
        case [*valcent, "sigma_absolute" | "sigma_relative" | "sigma_percent"]:
            match valcent:
                case ["value" | "central"] | ["value", "central"] | ["central", "value"]:
                    return True

    return False


def CheckCorrelationSizes(cfg):
    nnames = len(cfg["names"])
    matrix = cfg["matrix"]
    nrows = len(matrix)

    if nrows != nnames:
        return False

    return all(nnames == len(row) for row in matrix)


IsCorrelationsDict = And(
    {
        "names": Or((str,), And([str], Use(tuple))),
        "matrix_type": Or("correlation", "covariance"),
        "matrix": [[IsNumber]],
    },
    CheckCorrelationSizes,
)
IsNestedCorrelationsDict = NestedSchema(IsCorrelationsDict, processdicts=True)

IsFormat = Schema(IsFormatOk, error='Invalid parameter format "{}".')
IsParsCfgDict = Schema(
    {
        "parameters": IsValuesDict,
        "labels": IsLabelsDict,
        "format": IsFormat,
        "state": Or("fixed", "variable", error="Invalid parameters state: {}"),
        Optional("path", default=""): str,
        Optional("replicate", default=((),)): (IsStrSeqOrStr,),
        Optional("replica_key_offset", default=0): int,
        Optional("correlations", default={}): IsNestedCorrelationsDict,
        Optional("joint_nuisance", default=False): bool,
    },
    # error = 'Invalid parameters configuration: {}'
)
IsProperParsCfgDict = And(IsParsCfgDict, ParsCfgHasProperFormat())
IsLoadableDict = And(
    {"load": Or(str, And(Path, Use(str))), Optional(str): object},
    Use(
        LoadFileWithExt(yaml=LoadYaml, py=MakeLoaderPy("configuration"), key="load", update=True),
        error="Failed to load {}",
    ),
    IsProperParsCfgDict,
)


def ValidateParsCfg(cfg):
    if isinstance(cfg, dict) and "load" in cfg:
        return IsLoadableDict.validate(cfg)
    else:
        return IsProperParsCfgDict.validate(cfg)


def process_var_fixed1(vcfg, _, __):
    return {"central": vcfg, "value": vcfg, "sigma": None}


def process_var_fixed2(vcfg, format, hascentral) -> dict:
    ret = dict(zip(format, vcfg))
    if hascentral:
        ret.setdefault("value", ret["central"])
    else:
        ret.setdefault("central", ret["value"])
    ret["sigma"] = None
    return ret


def process_var_absolute(vcfg, format, hascentral) -> dict:
    ret = process_var_fixed2(vcfg, format, hascentral)
    ret["sigma"] = ret["sigma_absolute"]
    return ret


def process_var_relative(vcfg, format, hascentral) -> dict:
    ret = process_var_fixed2(vcfg, format, hascentral)
    ret["sigma"] = ret["sigma_relative"] * ret["central"]
    return ret


def process_var_percent(vcfg, format, hascentral) -> dict:
    ret = process_var_fixed2(vcfg, format, hascentral)
    ret["sigma"] = 0.01 * ret["sigma_percent"] * ret["central"]
    return ret


def get_format_processor(format):
    if isinstance(format, str):
        return process_var_fixed1

    errfmt = format[-1]
    if not errfmt.startswith("sigma"):
        return process_var_fixed2

    if errfmt.endswith("_absolute"):
        return process_var_absolute
    elif errfmt.endswith("_relative"):
        return process_var_relative
    else:
        return process_var_percent


def format_dict(dct: dict, /, *args, **kwargs) -> dict:
    return {k: format_latex(k, v, *args, **kwargs) for k, v in dct.items() if k in label_keys}


def get_label(key: tuple, labelscfg: dict) -> dict:
    try:
        ret = labelscfg.any(key)
    except KeyError:
        pass
    else:
        ret["index_values"] = list(key)
        if isinstance(ret, NestedMKDict):
            ret = ret.object
        return dict(ret)

    for n in range(1, len(key) + 1):
        subkey = key[:-n]
        try:
            lcfg = labelscfg.any(subkey)
        except KeyError:
            continue

        if not subkey and "text" not in lcfg:
            break

        key_str = ".".join(key[n - 1 :])
        ret = format_dict(
            lcfg, key_str, key=key_str, space_key=f" {key_str}", key_space=f"{key_str} "
        )
        ret["index_values"] = list(key)
        return ret

    return {}


def iterate_varcfgs(
    cfg: NestedMKDict,
) -> Generator[tuple[tuple[str, ...], NestedMKDict], None, None]:
    parameterscfg = cfg("parameters")
    labelscfg = cfg("labels")
    format = cfg["format"]

    hascentral = "central" in format
    process = get_format_processor(format)

    for key, varcfg in parameterscfg.walkitems():
        varcfg = process(varcfg, format, hascentral)
        varcfg["label"] = get_label(key, labelscfg)
        yield key, varcfg


from ..lib import Array
from ..lib.ElSumSq import ElSumSq
from ..parameters import Parameters


def check_correlations_consistent(cfg: NestedMKDict) -> None:
    parscfg = cfg("parameters")
    for key, corrcfg in cfg("correlations").walkdicts():
        # processed_cfgs.add(varcfg)
        names = corrcfg["names"]
        try:
            parcfg = parscfg(key)
        except KeyError:
            raise InitializationError(f"Failed to obtain parameters for {key}")

        inames = tuple(parcfg.walkjoinedkeys())
        if names != inames:
            raise InitializationError(
                f'Keys in {".".join(key)} are not consistent with names: {inames} and {names}'
            )


def load_parameters(
    acfg: OptionalType[Mapping] = None,
    *,
    nuisance_location: str | Sequence[str] | None = "statistic.nuisance.parts",
    **kwargs,
):
    acfg = dict(acfg or {}, **kwargs)
    cfg = ValidateParsCfg(acfg)
    cfg = NestedMKDict(cfg)

    pathstr = cfg["path"]
    path = tuple(pathstr.split(".")) if pathstr else ()

    state = cfg["state"]

    ret = NestedMKDict(
        {
            "parameter": {
                "all": {},
                "constant": {},
                "free": {},
                "constrained": {},
                "normalized": {},
            },
            "correlations": {},
            "stat": {
                "nuisance_parts": {},
                "nuisance": {},
            },
            "parameter_node": {
                "all": {},
                "constant": {},
                "free": {},
                "constrained": {}
            },
        },
        sep=".",
    )

    check_correlations_consistent(cfg)

    subkeys = cfg["replicate"]
    replica_key_offset = cfg["replica_key_offset"]
    if replica_key_offset > 0:
        make_key = lambda key, subkey: key[:-replica_key_offset] + subkey + key[replica_key_offset:]
    elif replica_key_offset == 0:
        make_key = lambda key, subkey: key + subkey
    else:
        raise ValueError(f"{replica_key_offset=} should be non-negative")

    varcfgs = NestedMKDict({})
    normpars = {}
    for key_general, varcfg in iterate_varcfgs(cfg):
        varcfg.setdefault(state, True)

        label_general = varcfg["label"]

        for subkey in subkeys:
            key = key_general + subkey
            key = make_key(key_general, subkey)
            key_str = ".".join(key)
            subkey_str = ".".join(subkey)

            label = format_dict(
                label_general.copy(),
                subkey=key_str,
                space_key=f" {subkey_str}",
                key_space=f"{subkey_str} ",
            )
            varcfg_sub = varcfg.copy()
            varcfg_sub["label"] = label
            label["paths"] = [key_str]
            label["index_values"] = key + subkey
            label.setdefault("text", key_str)

            varcfgs[key] = varcfg_sub

    processed_cfgs = set()
    pars = NestedMKDict({})
    for key, corrcfg in cfg("correlations").walkdicts():
        label = get_label(key + ("group",), cfg("labels"))

        matrixtype = corrcfg["matrix_type"]
        matrix = corrcfg["matrix"]
        mark_matrix = matrixtype == "correlation" and "C" or "V"
        label_mat = inherit_labels(
            label, fmtlong=f"{matrixtype.capitalize()} matrix: {{}}", fmtshort=mark_matrix + "({})"
        )
        label_mat["mark"] = mark_matrix
        label_mat = format_dict(label_mat, key="", space_key="", key_space="")
        matrix_array = Array("matrixtype", matrix, label=label_mat)

        for subkey in subkeys:
            fullkey = key + subkey
            subkey_str = ".".join(subkey)
            try:
                varcfg = varcfgs(fullkey)
            except KeyError:
                raise InitializationError(f"Failed to obtain parameters for {fullkey}")

            kwargs = {matrixtype: matrix_array}
            kwargs["value"] = (vvalue := [])
            kwargs["central"] = (vcentral := [])
            kwargs["sigma"] = (vsigma := [])
            kwargs["names"] = (names := [])
            for name, vcfg in varcfg.walkdicts(ignorekeys=("label",)):
                vvalue.append(vcfg["value"])
                vcentral.append(vcfg["central"])
                vsigma.append(vcfg["sigma"])
                names.append(name)
                processed_cfgs.add(fullkey + name)

            labelsub = format_dict(
                label, subkey=subkey_str, space_key=f" {subkey_str}", key_space=f"{subkey_str} "
            )
            labelsub["index_values"] = list(key + subkey)
            pars[fullkey] = Parameters.from_numbers(label=labelsub, **kwargs)

    for key, varcfg in varcfgs.walkdicts(ignorekeys=("label",)):
        if key in processed_cfgs:
            continue
        par = Parameters.from_numbers(**varcfg.object)
        pars[key] = par

    for key, par in pars.walkitems():
        pathkey = path + key
        if par.is_constrained:
            targetkey = ("constrained",) + pathkey
        elif par.is_fixed:
            targetkey = ("constant",) + pathkey
        else:
            targetkey = ("free",) + pathkey

        ret[("parameter_node",) + targetkey] = par
        ret[("parameter_node", "all") + pathkey] = par

        ptarget = ("parameter",) + targetkey
        atarget = ("parameter", "all") + pathkey
        for subname, subpar in par.iteritems():
            ret[ptarget + subname] = subpar
            ret[atarget + subname] = subpar

        if constraint := par.constraint:
            normpars_i = normpars.setdefault(key[0], [])
            normpars_i.append(constraint.normvalue_final)

            ntarget = ("parameter", "normalized", path) + key
            for subname, subpar in par.iteritems_norm():
                ret[ntarget + subname] = subpar

    joint_nuisance = cfg["joint_nuisance"]
    nuisance_location = properkey(nuisance_location, sep=".")
    if joint_nuisance:
        ssq = ElSumSq(f"nuisance: {pathstr}")
        for outputs in normpars.values():
            outputs >> ssq
        ssq.close()
        cpath = nuisance_location + (path,)
        ret[("nodes",) + cpath] = ssq
        ret[("outputs",) + cpath] = ssq.outputs[0]
    else:
        for name, outputs in normpars.items():
            ssq = ElSumSq(f"nuisance: {pathstr}.{name}")
            outputs >> ssq
            ssq.close()
            cpath = nuisance_location + (path, name)
            ret[("nodes",) + cpath] = ssq
            ret[("outputs",) + cpath] = ssq.outputs[0]

    NodeStorage.update_current(ret, strict=True)

    return ret
