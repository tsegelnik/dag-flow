from collections.abc import Sequence
from typing import Union

from numpy import result_type

from .exception import TypeFunctionError
from .types import NodeT


def check_has_inputs(node: NodeT) -> None:
    """Checking if the node has inputs"""
    if len(node.inputs) == 0:
        raise TypeFunctionError("Cannot use node with zero inputs!", node=node)


def eval_output_dtype(
    node: NodeT, outputkey: Union[str, int, slice, Sequence] = "result"
) -> None:
    """Automatic calculation and setting dtype for the output"""
    node.outputs[outputkey]._dtype = result_type(
        *tuple(inp.dtype for inp in node.inputs)
    )


def copy_input_dtype_to_output(
    node: NodeT,
    inputkey: Union[str, int, slice, Sequence] = 0,
    outputkey: Union[str, int, slice, Sequence] = "result",
) -> None:
    """Coping input dtype and setting for the output"""
    node.outputs[outputkey]._dtype = node.inputs[inputkey].dtype


def copy_input_shape_to_output(
    node: NodeT,
    inputkey: Union[str, int, slice, Sequence] = 0,
    outputkey: Union[str, int, slice, Sequence] = "result",
) -> None:
    """Coping input shape and setting for the output"""
    node.outputs[outputkey]._shape = node.inputs[inputkey].shape


def combine_inputs_shape_to_output(
    node: NodeT, outputkey: Union[str, int, slice, Sequence] = "result"
) -> None:
    """Combine all the inputs shape and setting for the output"""
    node.outputs[outputkey]._shape = tuple(inp.shape for inp in node.inputs)


def check_input_dimension(
    node: NodeT, inputkey: Union[str, int, slice, Sequence], ndim: int
):
    """Checking the dimension of the input"""
    input = node.inputs[inputkey]
    dim = len(input.shape)
    if dim != ndim:
        raise TypeFunctionError(
            f"The node supports only {ndim}d inputs, but given {dim}d!",
            node=node,
            input=input,
        )


def check_input_dtype(
    node: NodeT, inputkey: Union[str, int, slice, Sequence], dtype
):
    """Checking the dimension of the input"""
    input = node.inputs[inputkey]
    dtt = input.dtype
    if dtt != dtype:
        raise TypeFunctionError(
            f"The node supports only input types {dtype}, but given {dtt}!",
            node=node,
            input=input,
        )
