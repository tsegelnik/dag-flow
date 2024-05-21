from collections.abc import Callable, KeysView

from numpy.typing import NDArray

from dagflow.node import Node
from dagflow.output import Output
from dagflow.parameters import Parameter
from dagflow.storage import NestedMKDict, NodeStorage


def _return_data(node: Node | Output, copy: bool) -> NDArray | tuple[NDArray] | None:
    if isinstance(node, Output):
        return node.data.copy() if copy else node.data
    outputs = node.outputs
    if (nout := len(outputs)) == 0:
        return None
    elif nout == 1:
        return outputs[0].data.copy() if copy else outputs[0].data
    elif copy:
        return tuple(out.data.copy() for out in outputs)
    else:
        return tuple(out.data for out in outputs)


def _find_par_permissive(storage: NodeStorage | NestedMKDict, name: str) -> Parameter | None:
    for key, par in storage.walkitems():
        if key[-1] == name and isinstance(par, Parameter):
            return par


def _collect_pars_permissive(
    storage: NodeStorage | NestedMKDict, par_names: list[str] | tuple[str, ...] | KeysView
) -> dict[str, Parameter]:
    res = {}
    for name in par_names:
        if (par := _find_par_permissive(storage, name)) is not None:
            res[name] = par
    return res


def _get_parameter(
    name: str, parsdict: dict[str, Parameter], storage: NodeStorage | NestedMKDict
) -> Parameter:
    """
    Gets a parameter from the parameters dict,
    which stores the parameters found from the "fuzzy" search,
    or try to get the parameter from the storage,
    supposing that the name is the precise key in the storage
    """
    try:
        par = parsdict[name]
    except KeyError as exc:
        try:
            par = storage[name]
        except KeyError:
            raise RuntimeError(f"There is no parameter '{name}' in the {storage=}!") from exc
    return par


def makefcn(
    node: Node | Output,
    storage: NodeStorage | NestedMKDict,
    safe: bool = True,
    par_names: list[str] | tuple[str, ...] | None = None,
) -> Callable:
    """
    Retruns a function, which takes the parameter values as arguments
    and retruns the result of the node evaluation.

    :param node: A node (or output), depending (explicitly or implicitly) on the parameters
    :type node: class:`dagflow.node.Node` | class:`dagflow.output.Output`
    :param storage: A storage with parameters
    :type storage: class:`dagflow.storage.NodeStorage`
    :param safe: If `safe=True`, the parameters will be resetted to old values after evaluation.
    If `safe=False`, the parameters will be setted to the new values
    :type safe: bool
    :param par_names: The short names of the set of parameters for presearch
    :type par_names: list[str] | tuple[str] | None
    :rtype: function
    """
    if not isinstance(node, (Node, Output)):
        raise ValueError(f"`node` must be Node | Output, but given {node}, {type(node)=}!")
    if not isinstance(storage, (NodeStorage, NestedMKDict)):
        raise ValueError(
            f"`storage` must be NodeStorage | NestedMKDict, but given {storage}, {type(storage)=}!"
        )
    # the dict with parameters found from the presearch
    _parsdict = _collect_pars_permissive(storage, par_names) if par_names else {}

    match (_parsdict, safe):
        case ({}, False):
            def fcn_unsafe_runtime_search(**kwargs):
                parameters = _collect_pars_permissive(storage, kwargs.keys())
                for name, val in kwargs.items():
                    par = _get_parameter(name, parameters, storage)
                    par.value = val
                node.touch()
                return _return_data(node, copy=False)
            return fcn_unsafe_runtime_search
        case ({}, _):
            def fcn_safe_runtime_search(**kwargs):
                parameters = _collect_pars_permissive(storage, kwargs.keys())
                pars = []
                for name, val in kwargs.items():
                    par = _get_parameter(name, parameters, storage)
                    par.push(val)
                    pars.append(par)
                node.touch()
                res = _return_data(node, copy=True)
                for par in pars:
                    par.pop()
                node.touch()
                return res
            return fcn_safe_runtime_search
        case (_, False):
            def fcn_unsafe_presearch(**kwargs):
                for name, val in kwargs.items():
                    par = _get_parameter(name, _parsdict, storage)
                    par.value = val
                node.touch()
                return _return_data(node, copy=False)
            return fcn_unsafe_presearch
        case (_, _):
            def fcn_safe_presearch(**kwargs):
                pars = []
                for name, val in kwargs.items():
                    par = _get_parameter(name, _parsdict, storage)
                    par.push(val)
                    pars.append(par)
                node.touch()
                res = _return_data(node, copy=True)
                for par in pars:
                    par.pop()
                node.touch()
                return res
            return fcn_safe_presearch
