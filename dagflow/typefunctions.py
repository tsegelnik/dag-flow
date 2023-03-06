from collections.abc import Sequence
from typing import Union

from numpy import result_type
from itertools import repeat

from .exception import TypeFunctionError
from .types import NodeT

AllPositionals = slice(None)

try:
    zip((), (), strict=True)
except TypeError:
    # provide a replacement of strict zip from Python 3.1
    # to be deprecated at some point
    from itertools import zip_longest

    def zip(*iterables, strict: bool = False):
        sentinel = object()
        for combo in zip_longest(*iterables, fillvalue=sentinel):
            if strict and sentinel in combo:
                raise ValueError("Iterables have different lengths")
            yield combo


def check_has_inputs(
    node: NodeT, inputkey: Union[str, int, slice, Sequence, None] = None
) -> None:
    """Checking if the node has inputs"""
    if inputkey is None or inputkey == AllPositionals:
        try:
            node.inputs[0]
        except Exception as exc:
            raise TypeFunctionError(
                "The node must have at lease one input!", node=node
            ) from exc
    else:
        try:
            node.inputs[inputkey]
        except Exception as exc:
            raise TypeFunctionError(
                f"The node must have the input '{inputkey}'!", node=node
            ) from exc


def eval_output_dtype(
    node: NodeT,
    inputkey: Union[str, int, slice, Sequence] = AllPositionals,
    outputkey: Union[str, int, slice, Sequence] = AllPositionals,
) -> None:
    """Automatic calculation and setting dtype for the output"""
    inputs = node.inputs.iter(inputkey)
    outputs = node.outputs.iter(outputkey)

    dtype = result_type(*(inp.dd.dtype for inp in inputs))
    for output in outputs:
        output.dd.dtype = dtype


def copy_input_to_output(
    node: NodeT,
    inputkey: Union[str, int, slice, Sequence] = 0,
    outputkey: Union[str, int, slice, Sequence] = AllPositionals,
    dtype: bool = True,
    shape: bool = True,
) -> None:
    """Coping input dtype and setting for the output"""
    inputs = tuple(node.inputs.iter(inputkey))
    outputs = tuple(node.outputs.iter(outputkey))

    if dtype and shape:

        def cpy(input, output):
            output.dd.dtype = input.dd.dtype
            output.dd.shape = input.dd.shape

    elif dtype:

        def cpy(input, output):
            output.dd.dtype = input.dd.dtype

    elif shape:

        def cpy(input, output):
            output.dd.shape = input.dd.shape

    else:
        return

    if len(inputs) == 1:
        inputs = repeat(inputs[0], len(outputs))

    for input, output in zip(inputs, outputs, strict=True):
        cpy(input, output)


def copy_input_dtype_to_output(
    node: NodeT,
    inputkey: Union[str, int, slice, Sequence] = 0,
    outputkey: Union[str, int, slice, Sequence] = AllPositionals,
) -> None:
    """Coping input dtype and setting for the output"""
    inputs = tuple(node.inputs.iter(inputkey))
    outputs = tuple(node.outputs.iter(outputkey))

    if len(inputs) == 1:
        inputs = repeat(inputs[0], len(outputs))

    for input, output in zip(inputs, outputs, strict=True):
        output.dd.dtype = input.dd.dtype


def copy_input_shape_to_output(
    node: NodeT,
    inputkey: Union[str, int] = 0,
    outputkey: Union[str, int, slice, Sequence] = AllPositionals,
) -> None:
    """Coping input shape and setting for the output"""
    inputs = tuple(node.inputs.iter(inputkey))
    outputs = tuple(node.outputs.iter(outputkey))

    if len(inputs) == 1:
        inputs = repeat(inputs[0], len(outputs))

    for input, output in zip(inputs, outputs, strict=True):
        output.dd.shape = input.dd.shape


def combine_inputs_shape_to_output(
    node: NodeT,
    inputkey: Union[str, int, slice, Sequence] = AllPositionals,
    outputkey: Union[str, int, slice, Sequence] = AllPositionals,
) -> None:
    """Combine all the inputs shape and setting for the output"""
    inputs = node.inputs.iter(inputkey)
    shape = tuple(inp.dd.shape for inp in inputs)
    for output in node.outputs.iter(outputkey):
        output.dd.shape = shape


def check_input_dimension(
    node: NodeT, inputkey: Union[str, int, slice, Sequence], ndim: int
):
    """Checking the dimension of the input"""
    for input in node.inputs.iter(inputkey):
        dim = len(input.dd.shape)
        if ndim != dim:
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
        shape = input.dd.shape
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
        shape = input.dd.shape
        dim = len(shape)
        dim_max = max(dim, dim_max)
        if (dim == 2 and shape[0] != shape[1]) and dim != 1:
            raise TypeFunctionError(
                f"The node supports only square inputs (or 1d as diagonal). Got {shape}!",
                node=node,
                input=input,
            )
    return dim_max

def check_input_shape(
    node: NodeT, inputkey: Union[str, int, slice, Sequence], shape: tuple
):
    """Checking the shape equivalence for inputs"""
    for input in node.inputs.iter(inputkey):
        sshape = input.dd.shape
        if sshape != shape:
            raise TypeFunctionError(
                f"The node supports only inputs with shape={shape}. Got {sshape}!",
                node=node,
                input=input,
            )


def check_input_dtype(
    node: NodeT, inputkey: Union[str, int, slice, Sequence], dtype
):
    """Checking the dtype equivalence for inputs"""
    for input in node.inputs.iter(inputkey):
        dtt = input.dd.dtype
        if dtt != dtype:
            raise TypeFunctionError(
                f"The node supports only input types {dtype}. Got {dtt}!",
                node=node,
                input=input,
            )


def check_inputs_equivalence(
    node: NodeT, inputkey: Union[str, int, slice, Sequence] = AllPositionals
):
    """Checking the equivalence of the dtype and shape of all the inputs"""
    inputs = tuple(node.inputs.iter(inputkey))
    input0, inputs = inputs[0], inputs[1:]

    dtype, shape = input0.dd.dtype, input0.dd.shape
    for input in inputs:
        if input.dd.dtype != dtype or input.dd.shape != shape:
            raise TypeFunctionError(
                f"Input data {input.dtype} [{input.shape}] is inconsistent with {dtype} [{shape}]",
                node=node,
                input=input,
            )

def check_inputs_square_or_diag(
    node: NodeT,
    inputkey: Union[str, int, slice, Sequence] = AllPositionals,
) -> int:
    """Check if inputs are square matrices or diagonals (1d) of a square matrices of the same size.
    Returns the maximal dimension."""
    inputs = tuple(node.inputs.iter(inputkey))

    dim_max = 0
    shape0 = inputs[0].dd.shape[0]

    for input in inputs:
        shape = input.dd.shape
        dim = len(shape)
        dim_max = max(dim, dim_max)
        if shape0 != shape[0] or ((dim == 2 and shape[0] != shape[1]) and dim != 1):
            raise TypeFunctionError(
                f"The node supports only square inputs (or 1d as diagonal) of size {shape0}x{shape0}. Got {shape}!",
                node=node,
                input=input,
            )
    return dim_max


def check_inputs_same_dtype(
    node: NodeT, inputkey: Union[str, int, slice, Sequence] = AllPositionals
):
    """Checking dtypes of all the inputs are same"""
    inputs = tuple(node.inputs.iter(inputkey))
    input0, inputs = inputs[0], inputs[1:]

    dtype = input0.dd.dtype
    for input in inputs:
        if input.dd.dtype != dtype:
            raise TypeFunctionError(
                f"Input data {input.dd.dtype} is inconsistent with {dtype}",
                node=node,
                input=input,
            )


def check_inputs_multiplicable_mat(
    node: NodeT,
    inputkey1: Union[str, int, slice, Sequence],
    inputkey2: Union[str, int, slice, Sequence],
):
    """Checking that inputs from key1 and key2 may be multiplied (matrix)"""
    inputs1 = tuple(node.inputs.iter(inputkey1))
    inputs2 = tuple(node.inputs.iter(inputkey2))

    len1, len2 = len(inputs1), len(inputs2)
    if len1 == len2:
        pass
    elif len1 == 1:
        inputs1 = repeat(inputs1[0], len2)
    elif len2 == 1:
        inputs2 = repeat(inputs2[0], len1)

    for input1, input2 in zip(inputs1, inputs2, strict=True):
        shape1 = input1.dd.shape
        shape2 = input2.dd.shape
        if shape1[-1] != shape2[0]:
            raise TypeFunctionError(
                f"Inputs {shape1} and {shape2} are not multiplicable",
                node=node,
                input=input,
            )
