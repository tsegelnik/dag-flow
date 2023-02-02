from collections.abc import Sequence
from typing import Union

from numpy import result_type

from dagflow.tools import IsIterable

from .exception import TypeFunctionError
from .types import NodeT

def check_has_inputs(node: NodeT) -> None:
    """Checking if the node has inputs"""
    if len(node.inputs) == 0:
        raise TypeFunctionError("Cannot use node with zero inputs!", node=node)


def eval_output_dtype(
    node: NodeT,
    inputkey: Union[str, int, slice, Sequence] = slice(None),
    outputkey: Union[str, int, slice, Sequence] = slice(None)
) -> None:
    """Automatic calculation and setting dtype for the output"""
    inputs = node.inputs.iter(inputkey)
    outputs = node.outputs.iter(outputkey)

    dtype = result_type(*(inp.dtype for inp in inputs))
    for output in outputs:
        output._dtype = dtype


def copy_input_to_output(
    node: NodeT,
    inputkey: Union[str, int, slice, Sequence] = 0,
    outputkey: Union[str, int, slice, Sequence] = slice(None),
    dtype: bool = True, shape: bool = True
) -> None:
    """Coping input dtype and setting for the output"""
    inputs = tuple(node.inputs.iter(inputkey))
    outputs = tuple(node.outputs.iter(outputkey))

    if dtype and shape:
        def cpy(input, output):
            output._dtype = input.dtype
            output._shape = input.shape
    elif dtype:
        def cpy(input, output):
            output._dtype = input.dtype
    elif shape:
        def cpy(input, output):
            output._shape = input.shape
    else:
        return

    if len(inputs)==1:
        input0 = inputs[0]
        for output in outputs:
            cpy(input0, output)
    else:
        for input, output in zip(inputs, outputs):
            cpy(input, output)


def copy_input_dtype_to_output(
    node: NodeT,
    inputkey: Union[str, int, slice, Sequence] = 0,
    outputkey: Union[str, int, slice, Sequence] = slice(None)
) -> None:
    """Coping input dtype and setting for the output"""
    inputs = tuple(node.inputs.iter(inputkey))
    outputs = node.outputs.iter(outputkey)

    if len(inputs)==1:
        dtype = inputs[0].dtype
        for output in outputs:
            output._dtype = dtype
    else:
        for input, output in zip(inputs, outputs):
            output._dtype = input.dtype


def copy_input_shape_to_output(
    node: NodeT,
    inputkey: Union[str, int] = 0,
    outputkey: Union[str, int, slice, Sequence] = slice(None)
) -> None:
    """Coping input shape and setting for the output"""
    inputs = tuple(node.inputs.iter(inputkey))
    outputs = node.outputs.iter(outputkey)

    if len(inputs)==1:
        shape = inputs[0].shape
        for output in outputs:
            output._shape = shape
    else:
        for input, output in zip(inputs, outputs):
            output._shape = input.shape


def combine_inputs_shape_to_output(
    node: NodeT,
    inputkey: Union[str, int, slice, Sequence] = slice(None),
    outputkey: Union[str, int, slice, Sequence] = slice(None)
) -> None:
    """Combine all the inputs shape and setting for the output"""
    inputs = node.inputs.iter(inputkey)
    shape = tuple(inp.shape for inp in inputs)
    for output in node.outputs.iter(outputkey):
        output._shape = shape


def check_input_dimension(
    node: NodeT,
    inputkey: Union[str, int, slice, Sequence],
    ndim: int
):
    """Checking the dimension of the input"""
    for input in node.inputs[inputkey]:
        dim = len(input.shape)
        if dim != ndim:
            raise TypeFunctionError(
                f"The node supports only {ndim}d inputs, but given {dim}d!",
                node=node,
                input=input,
            )


def check_input_dtype(
    node: NodeT,
    inputkey: Union[str, int, slice, Sequence],
    dtype
):
    """Checking the dimension of the input"""
    for input in node.inputs[inputkey]:
        dtt = input.dtype
        if dtt != dtype:
            raise TypeFunctionError(
                f"The node supports only input types {dtype}, but given {dtt}!",
                node=node,
                input=input,
            )


def check_inputs_equivalence(
    node: NodeT,
    inputkey: Union[str, int, slice, Sequence] = slice(None)
):
    """Checking the equivalence of the dtype and shape of all the inputs"""
    inputs = tuple(node.inputs.iter(inputkey))
    input0, inputs = inputs[0], inputs[1:]

    dtype, shape = input0.dtype, input0.shape
    for input in inputs:
        if input.dtype != dtype or input.shape != shape:
            raise TypeFunctionError(
                f"Input data {input.dtype} [{input.shape}] is inconsistent with {dtype} [{shape}]",
                node=node,
                input=input,
            )
