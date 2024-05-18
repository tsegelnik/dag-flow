from dagflow.node import Node
from dagflow.parameters import Parameter
from dagflow.storage import NodeStorage


def _return_data(node: Node, copy: bool):
    outputs = node.outputs
    if (nout := len(outputs)) == 0:
        return None
    elif nout == 1:
        return outputs[0].data.copy() if copy else outputs[0].data
    else:
        return (out.data.copy() for out in outputs) if copy else (out.data for out in outputs)


def makefcn(node: Node, storage: NodeStorage, safe: bool = True):
    # TODO: update search of parameters
    def fcn_safe(**kwargs):
        parameters = storage("parameter.all")
        pars = []
        for name, val in kwargs.items():
            par = parameters[name]
            if not isinstance(par, Parameter):
                raise RuntimeError(f"Cannot find a patameter with {name=} in the {storage=}")
            par.push(val)
            pars.append(par)
        node.touch()
        res = _return_data(node, copy=True)
        for par in pars:
            par.pop()
        node.touch()
        return res

    def fcn_nonsafe(**kwargs):
        # TODO: update search of parameters
        parameters = storage("parameter.all")
        for name, val in kwargs.items():
            par = parameters[name]
            if not isinstance(par, Parameter):
                raise RuntimeError(f"Cannot find a patameter with {name=} in the {storage=}")
            par.value = val
        node.touch()
        return _return_data(node, copy=False)

    return fcn_safe if safe else fcn_nonsafe
