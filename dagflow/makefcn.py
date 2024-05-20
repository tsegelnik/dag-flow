from collections.abc import Callable, KeysView

from dagflow.node import Node
from dagflow.parameters import Parameter
from dagflow.storage import NestedMKDict, NodeStorage


def _return_data(node: Node, copy: bool):
    outputs = node.outputs
    if (nout := len(outputs)) == 0:
        return None
    elif nout == 1:
        return outputs[0].data.copy() if copy else outputs[0].data
    else:
        return (out.data.copy() for out in outputs) if copy else (out.data for out in outputs)


def _find_par_permissive(storage: NodeStorage | NestedMKDict, name: str) -> Parameter:
    for key, par in storage.walkitems():
        if key[-1] == name and isinstance(par, Parameter):
            return par
    raise RuntimeError(f"Cannot find the parameter '{name}' in the {storage=}!")


def _collect_pars_permissive(
    storage: NodeStorage | NestedMKDict, par_names: list[str] | tuple[str, ...] | KeysView
) -> dict[str, Parameter]:
    return {name: _find_par_permissive(storage, name) for name in par_names}


def makefcn(
    node: Node,
    storage: NodeStorage | NodeStorage,
    safe: bool = True,
    par_names: list[str] | tuple[str, ...] | None = None,
) -> Callable:
    """
    Retruns a function, which takes the parameter values as arguments
    and retruns the result of the node evaluation.

    :param node: A node, depending (explicitly or implicitly) on the parameters
    :type node: class:`dagflow.node.Node`
    :param storage: A storage with parameters
    :type storage: class:`dagflow.storage.NodeStorage`
    :param safe: If `safe=True`, the parameters will be resetted to old values after evaluation.
    If `safe=False`, the parameters will be setted to the new values
    :type safe: bool
    :param par_names: The names of parameters for changing
    :type par_names: list[str] | tuple[str] | None
    :rtype: function
    """
    if not isinstance(storage, (NodeStorage, NestedMKDict)):
        raise ValueError(
            f"storage must be NodeStorage | NestedMKDict, but given {storage}, {type(storage)=}!"
        )
    parsdict = _collect_pars_permissive(storage, par_names) if par_names else {}

    def fcn_safe(**kwargs):
        pars = []
        for name, val in kwargs.items():
            try:
                par = parsdict[name]
            except KeyError:
                raise RuntimeError(
                    f"There is no parameter '{name}'! Allowed parameters are {parsdict.keys()}"
                )
            par.push(val)
            pars.append(par)
        node.touch()
        res = _return_data(node, copy=True)
        for par in pars:
            par.pop()
        node.touch()
        return res

    def fcn_nonsafe(**kwargs):
        for name, val in kwargs.items():
            try:
                par = parsdict[name]
            except KeyError:
                raise RuntimeError(
                    f"There is no parameter '{name}'! Allowed parameters are {parsdict.keys()}"
                )
            par.value = val
        node.touch()
        return _return_data(node, copy=False)

    def fcn_safe_with_search(**kwargs):
        parameters = _collect_pars_permissive(storage, kwargs.keys())
        for name, val in kwargs.items():
            par = parameters[name]
            par.push(val)
        node.touch()
        res = _return_data(node, copy=True)
        for par in parameters.values():
            par.pop()
        node.touch()
        return res

    def fcn_nonsafe_with_search(**kwargs):
        parameters = _collect_pars_permissive(storage, kwargs.keys())
        for name, val in kwargs.items():
            par = parameters[name]
            par.value = val
        node.touch()
        return _return_data(node, copy=False)

    if parsdict:
        return fcn_safe if safe else fcn_nonsafe
    return fcn_safe_with_search if safe else fcn_nonsafe_with_search
