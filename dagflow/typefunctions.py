from collections.abc import Sequence
from typing import Union

from numpy import result_type
from itertools import repeat

from .exception import TypeFunctionError
from .types import NodeT

try:
    zip((), (), strict=True)
except TypeError:
    # provide a replacement of strict zip from Python 3.1
    # to be deprecated at some point
    from itertools import zip_longest
    def zip(*iterables, strict: bool=False):
        sentinel = object()
        for combo in zip_longest(*iterables, fillvalue=sentinel):
            if strict and sentinel in combo:
                raise ValueError('Iterables have different lengths')
            yield combo

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
        inputs = repeat(inputs[0], len(outputs))

    for input, output in zip(inputs, outputs, strict=True):
        cpy(input, output)


def copy_input_dtype_to_output(
    node: NodeT,
    inputkey: Union[str, int, slice, Sequence] = 0,
    outputkey: Union[str, int, slice, Sequence] = slice(None)
) -> None:
    """Coping input dtype and setting for the output"""
    inputs = tuple(node.inputs.iter(inputkey))
    outputs = tuple(node.outputs.iter(outputkey))

    if len(inputs)==1:
        inputs = repeat(inputs[0], len(outputs))

    for input, output in zip(inputs, outputs, strict=True):
        output._dtype = input.dtype


def copy_input_shape_to_output(
    node: NodeT,
    inputkey: Union[str, int] = 0,
    outputkey: Union[str, int, slice, Sequence] = slice(None)
) -> None:
    """Coping input shape and setting for the output"""
    inputs = tuple(node.inputs.iter(inputkey))
    outputs = tuple(node.outputs.iter(outputkey))

    if len(inputs)==1:
        inputs = repeat(inputs[0], len(outputs))

    for input, output in zip(inputs, outputs, strict=True):
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
    ndim: int,
    strict: bool = True,
):
    """Checking the dimension of the input"""
    for input in node.inputs.iter(inputkey):
        dim = len(input.shape)
        if ndim != dim or (
            not strict and ndim != dim - 1 and input.shape[0] != 1
        ):
            raise TypeFunctionError(
                f"The node supports only {ndim}d inputs. Got {dim}d!",
                node=node,
                input=input,
            )

def check_input_square(
    node: NodeT,
    inputkey: Union[str, int, slice, Sequence],
):
    """Checking input is a square matrix"""
    for input in node.inputs.iter(inputkey):
        shape = input.shape
        dim = len(shape)
        if dim != 2 or shape[0] != shape[1]:
            raise TypeFunctionError(
                f"The node supports only square inputs. Got {shape}!",
                node=node,
                input=input,
            )

def check_input_square_or_diag(
    node: NodeT,
    inputkey: Union[str, int, slice, Sequence],
) -> int:
    """Check if input is a square matrix or diagonal (1d) of a square matrix.
    Returns the maximal dimension."""
    dim_max = 0
    for input in node.inputs.iter(inputkey):
        shape = input.shape
        dim = len(shape)
        dim_max = max(dim, dim_max)
        if (dim == 2 and shape[0] != shape[1]) and dim != 1:
            raise TypeFunctionError(
                f"The node supports only square inputs (or 1d as diagonal). Got {shape}!",
                node=node,
                input=input,
            )
    return dim_max

def check_input_dtype(
    node: NodeT,
    inputkey: Union[str, int, slice, Sequence],
    dtype
):
    """Checking the dimension of the input"""
    for input in node.inputs.iter(inputkey):
        dtt = input.dtype
        if dtt != dtype:
            raise TypeFunctionError(
                f"The node supports only input types {dtype}. Got {dtt}!",
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

def check_inputs_multiplicable_mat(
    node: NodeT,
    inputkey1: Union[str, int, slice, Sequence],
    inputkey2: Union[str, int, slice, Sequence]
):
    """Checking that inputs from key1 and key2 may be multiplied (matrix)"""
    inputs1 = tuple(node.inputs.iter(inputkey1))
    inputs2 = tuple(node.inputs.iter(inputkey2))

    len1, len2 = len(inputs1), len(inputs2)
    if len1==len2:
        pass
    elif len1==1:
        inputs1 = repeat(inputs1[0], len2)
    elif len2==1:
        inputs2 = repeat(inputs2[0], len1)

    for input1, input2 in zip(inputs1, inputs2, strict=True):
        shape1 = input1.shape
        shape2 = input2.shape
        if shape1[-1]!=shape2[0]:
            raise TypeFunctionError(
                f"Inputs {shape1} and {shape2} are not multiplicable",
                node=node,
                input=input,
            )
