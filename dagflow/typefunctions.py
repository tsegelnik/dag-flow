from collections.abc import Sequence
from itertools import repeat
from typing import Optional, Tuple, Union

from numpy import issubdtype, result_type
from numpy.typing import DTypeLike

from .exception import TypeFunctionError
from .input import Input
from .output import Output
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


class MethodSequenceCaller:
    """Class to call a sequence of methods"""

    methods: list

    def __init__(self) -> None:
        self.methods = []

    def __call__(self, inputs, outputs):
        for method in self.methods:
            method(inputs, outputs)


def cpy_dtype(input, output):
    output.dd.dtype = input.dd.dtype


def cpy_shape(input, output):
    output.dd.shape = input.dd.shape


def cpy_edges(input, output):
    output.dd.axes_edges = input.dd.axes_edges


def cpy_nodes(input, output):
    output.dd.axes_nodes = input.dd.axes_nodes


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


def check_inputs_number(node: NodeT, n: int) -> None:
    """Checking if the node has only `n` inputs"""
    if (ninp := len(node.inputs)) != n:
        raise TypeFunctionError(
            f"The node must have only {n} inputs, but given {ninp}!", node=node
        )


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
    edges: bool = True,
    nodes: bool = True,
) -> None:
    """Coping input dtype and setting for the output"""
    inputs = tuple(node.inputs.iter(inputkey))
    outputs = tuple(node.outputs.iter(outputkey))

    if not any((dtype, shape, edges, nodes)):
        return

    caller = MethodSequenceCaller()
    if dtype:
        caller.methods.append(cpy_dtype)
    if shape:
        caller.methods.append(cpy_shape)
    if edges:
        caller.methods.append(cpy_edges)
    if nodes:
        caller.methods.append(cpy_nodes)

    if len(inputs) == 1:
        inputs = repeat(inputs[0], len(outputs))

    for input, output in zip(inputs, outputs, strict=True):
        caller(input, output)


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


def copy_input_edges_to_output(
    node: NodeT,
    inputkey: Union[str, int] = 0,
    outputkey: Union[str, int, slice, Sequence] = AllPositionals,
) -> None:
    """Coping input edges and setting for the output"""
    inputs = tuple(node.inputs.iter(inputkey))
    outputs = tuple(node.outputs.iter(outputkey))

    if len(inputs) == 1:
        inputs = repeat(inputs[0], len(outputs))

    for input, output in zip(inputs, outputs, strict=True):
        output.dd.axes_edges = input.dd.axes_edges


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
    """Checking the equivalence of the dtype, shape, axes_edges and axes_nodes of all the inputs"""
    inputs = tuple(node.inputs.iter(inputkey))
    input0, inputs = inputs[0], inputs[1:]

    dtype, shape, edges, nodes = (
        input0.dd.dtype,
        input0.dd.shape,
        input0.dd.axes_edges,
        input0.dd.axes_nodes,
    )
    for input in inputs:
        if (
            input.dd.dtype != dtype
            or input.dd.shape != shape
            or input.dd.axes_edges != edges
            or input.dd.axes_nodes != nodes
        ):
            raise TypeFunctionError(
                f"Input data [{input.dtype=}, {input.shape=}, {input.axes_edges=}, {input.axes_nodes=}]"
                f" is inconsistent with [{dtype=}, {shape=}, {edges=}, {nodes=}]",
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
        if shape0 != shape[0] or (
            (dim == 2 and shape[0] != shape[1]) and dim != 1
        ):
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


def check_input_subtype(node: NodeT, input: Input, dtype: DTypeLike):
    """Checks if the input dtype is some subtype of `dtype`."""
    if not issubdtype(input.dd.dtype, dtype):
        raise TypeFunctionError(
            f"The input must be an array of {dtype}, but given '{input.dd.dtype}'!",
            node=node,
            input=input,
        )


def check_output_subtype(node: NodeT, output: Output, dtype: DTypeLike):
    """Checks if the output dtype is some subtype of `dtype`."""
    if not issubdtype(output.dd.dtype, dtype):
        raise TypeFunctionError(
            f"The output must be an array of {dtype}, but given '{output.dd.dtype}'!",
            node=node,
            output=output,
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


def check_input_edges_dim(
    node: NodeT,
    inputkey: Union[str, int, slice, Sequence] = AllPositionals,
    dim: int = 1,
):
    """Checking the existence and dim of the edges of the inputs"""
    for input in node.inputs.iter(inputkey):
        edges = input.dd.axes_edges
        if len(edges) == 0:
            raise TypeFunctionError(
                f"The input must have edges, but given {edges=}!",
                node=node,
                input=input,
            )
        for edge in edges:
            if not isinstance(edge, Output):
                raise TypeFunctionError(
                    f"The input edge must be an `Output`, but given {edge=}!",
                    node=node,
                    input=input,
                )
            if edge.dd.dim != dim:
                raise TypeFunctionError(
                    f"The input edge must be a {dim}d array, but given {edge.dd.dim=}!",
                    node=node,
                    input=input,
                )


def check_input_edges_equivalence(
    node: NodeT,
    inputkey: Union[str, int, slice, Sequence] = AllPositionals,
    reference: Optional[Tuple[Output]] = None,
):
    """Checking the equivalence of the edges of the inputs."""
    inputs = tuple(node.inputs.iter(inputkey))
    if reference is None:
        input0, inputs = inputs[0], inputs[1:]
        reference = input0.dd.axes_edges
    for input in inputs:
        edges = input.dd.axes_edges
        if edges != reference:
            raise TypeFunctionError(
                f"The input edge must be {reference}, but given {edges=}!",
                node=node,
                input=input,
            )


def check_edges_type(
    node: NodeT,
    inputkey: Union[str, int, slice, Sequence] = AllPositionals,
    outputkey: Union[str, int, slice, Sequence] = AllPositionals,
):
    """Checking of the edges type (must be `List[Output]`) of the inputs and outputs."""
    # check inputs
    for input in node.inputs.iter(inputkey):
        edges = input.dd.axes_edges
        if not isinstance(edges, list):
            raise TypeFunctionError(
                f"The `input.dd.axes_edges` must be `List[Output]`, but given {edges=}!",
                node=node,
                input=input,
            )
        for edge in edges:
            if not isinstance(edge, Output):
                raise TypeFunctionError(
                    f"The edge must be `Output`, but given {edge=}!",
                    node=node,
                    input=input,
                )
    # check outputs
    for output in node.outputs.iter(outputkey):
        edges = output.dd.axes_edges
        if not isinstance(edges, list):
            raise TypeFunctionError(
                f"The `output.dd.axes_edges` must be `List[Output]`, but given {edges=}!",
                node=node,
                output=output,
            )
        for edge in edges:
            if not isinstance(edge, Output):
                raise TypeFunctionError(
                    f"The edge must be `Output`, but given {edge=}!",
                    node=node,
                    iutput=output,
                )


def check_array_edges_consistency(node: NodeT, output: Output):
    """
    Checks the dimension equivalence of edges and the output, then checks that
    `len(output) = N` and `len(edges) = N+1` for each dimension.
    Tht type function is passed if the edges are empty.
    """
    dd = output.dd
    edges = dd.axes_edges
    if (y := len(edges)) > 0:
        if y != dd.dim:
            raise TypeFunctionError(
                f"Array: the data ({dd.dim}d) and edges "
                f"({len(edges)}d) must have the same dimension!",
                node=node,
                output=output,
            )
        for i, edge in enumerate(edges):
            if edge.dd.shape[0] != dd.shape[i] + 1:
                raise TypeFunctionError(
                    f"Array: the data lenght (={dd.shape[i]} + 1) must be "
                    f"consistent with edges (={edge.dd.shape[0]})!",
                    node=node,
                    output=output,
                )
