from itertools import repeat
from typing import Callable, Optional, Tuple, Union, Sequence

from numpy import issubdtype, result_type
from numpy.typing import DTypeLike

from .exception import TypeFunctionError
from .output import Output
from .input import Input
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

    __slots__ = ("methods",)

    def __init__(self) -> None:
        self.methods = []

    def __call__(self, inputs, outputs) -> None:
        for method in self.methods:
            method(inputs, outputs)

    def add(self, method: Callable) -> None:
        self.methods.append(method)


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

def copy_from_input_to_output(
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
        caller.add(cpy_dtype)
    if shape:
        caller.add(cpy_shape)
    if edges:
        caller.add(cpy_edges)
    if nodes:
        caller.add(cpy_nodes)

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


def check_input_dimension(
    node: NodeT, inputkey: Union[str, int, slice, Sequence], ndim: int, **kwargs
):
    """Checking the dimension of the input"""
    for input in node.inputs.iter(inputkey, **kwargs):
        dim = len(input.dd.shape)
        if ndim != dim:
            raise TypeFunctionError(
                f"The node supports only {ndim}d inputs. Got {dim}d!",
                node=node,
                input=input,
            )


def check_input_shape(
    node: NodeT, inputkey: Union[str, int, slice, Sequence], shape: tuple, **kwargs
):
    """Checking the shape equivalence for inputs"""
    for input in node.inputs.iter(inputkey, **kwargs):
        sshape = input.dd.shape
        if sshape != shape:
            raise TypeFunctionError(
                f"The node supports only inputs with shape={shape}. Got {sshape}!",
                node=node,
                input=input,
            )


def check_input_dtype(
    node: NodeT, inputkey: Union[str, int, slice, Sequence], dtype, **kwargs
):
    """Checking the dtype equivalence for inputs"""
    for input in node.inputs.iter(inputkey, **kwargs):
        dtt = input.dd.dtype
        if dtt != dtype:
            raise TypeFunctionError(
                f"The node supports only input types {dtype}. Got {dtt}!",
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
        if dim > 2:
            raise TypeFunctionError(
                f"The node supports only 1d or 2d. Got {dim}d!",
                node=node,
                input=input,
            )
        if (dim == 2 and shape[0] != shape[1]) and dim != 1:
            raise TypeFunctionError(
                f"The node supports only square inputs (or 1d as diagonal). Got {shape}!",
                node=node,
                input=input,
            )
    return dim_max


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
        if dim > 2:
            raise TypeFunctionError(
                f"The node supports only 1d or 2d. Got {dim}d!",
                node=node,
                input=input,
            )
        if shape0 != shape[0] or (
            (dim == 2 and shape[0] != shape[1]) and dim != 1
        ):
            raise TypeFunctionError(
                f"The node supports only square inputs (or 1d as diagonal) of size {shape0}x{shape0}. Got {shape}!",
                node=node,
                input=input,
            )
    return dim_max


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
        dd = input.dd
        if (
            dd.dtype != dtype
            or dd.shape != shape
            or (dd.axes_edges and edges and dd.axes_edges != edges)
            or (dd.axes_nodes and nodes and dd.axes_nodes != nodes)
        ):
            raise TypeFunctionError(
                f"Input data [{dd.dtype=}, {dd.shape=}, {dd.axes_edges=}, {dd.axes_nodes=}]"
                f" is inconsistent with [{dtype=}, {shape=}, {edges=}, {nodes=}]",
                node=node,
                input=input,
            )


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


def check_inputs_same_shape(
    node: NodeT, inputkey: Union[str, int, slice, Sequence] = AllPositionals
):
    """Checking shapes of all the inputs are same"""
    inputs = tuple(node.inputs.iter(inputkey))
    input0, inputs = inputs[0], inputs[1:]

    shape = input0.dd.shape
    for input in inputs:
        if input.dd.shape != shape:
            raise TypeFunctionError(
                f"Input data {input.dd.shape} is inconsistent with {shape}",
                node=node,
                input=input,
            )


def check_input_subtype(
    node: NodeT, inputkey: Union[str, int, slice, Sequence], dtype: DTypeLike
):
    """Checks if the input dtype is some subtype of `dtype`."""
    for input in node.inputs.iter(inputkey):
        if not issubdtype(input.dd.dtype, dtype):
            raise TypeFunctionError(
                f"The input must be an array of {dtype}, but given '{input.dd.dtype}'!",
                node=node,
                input=input,
            )


def check_output_subtype(
    node: NodeT, outputkey: Union[str, int, slice, Sequence], dtype: DTypeLike
):
    """Checks if the output dtype is some subtype of `dtype`."""
    for output in node.outputs.iter(outputkey):
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

def assign_output_edges(input: Union[Input, Sequence[Input]], output: Output, ignore_assigned: bool = False):
    """Assign output's edges from input's parent output"""
    dd = output.dd

    if dd.axes_edges:
        if ignore_assigned:
            return
        raise TypeFunctionError("Edges already assigned", output=output, input=input)

    if isinstance(input, Input):
        edges = [input.parent_output]
    else:
        edges = [inp.parent_output for inp in input]

    if len(dd.shape)!=len(edges):
        raise TypeFunctionError(
            f"Output ndim={len(dd.shape)} is inconsistent with edges ndim={len(edges)}",
            output=output,
            input=input
        )

    for i, (dimsize, edgeoutput) in enumerate(zip(dd.shape, edges)):
        shapeedges = edgeoutput.dd.shape
        if len(shapeedges)!=1:
            raise TypeFunctionError(
                    f"Edges of {i}th dimension has non-1d shape",
                    output=output,
                    input=input
                    )
        dimedges = shapeedges[0]
        if dimsize!=(dimedges-1):
            raise TypeFunctionError(
                f"Output dimension {i} size={dimsize} is inconsistent with edges size={len(dimedges)}",
                output=output,
                input=input
            )

    output.dd.axes_edges = edges

def assign_output_nodes(input: Union[Input, Sequence[Input]], output: Output, *, ignore_assigned: bool = False):
    """Assign output's edges from input's parent output"""
    dd = output.dd

    if dd.axes_nodes:
        if ignore_assigned:
            return
        raise TypeFunctionError("Nodes already assigned", output=output, input=input)

    if isinstance(input, Input):
        nodes = [input.parent_output]
    else:
        nodes = [inp.parent_output for inp in input]

    if len(dd.shape)!=len(nodes):
        raise TypeFunctionError(
            f"Output ndim={len(dd.shape)} is inconsistent with nodes ndim={len(nodes)}",
            output=output,
            input=input
        )

    for i, nodesoutput in enumerate(nodes):
        if dd.shape!=nodesoutput.dd.shape:
            raise TypeFunctionError(
                f"Output shape={dd.shape} is inconsistent with nodes {i} shape={nodesoutput.dd.shape}",
                output=output,
                input=input
            )

    output.dd.axes_nodes = nodes

def assign_output_axes_from_inputs(
    node: NodeT,
    inputkey: Union[str, int, slice, Sequence] = 0,
    outputkey: Union[str, int, slice, Sequence] = AllPositionals,
    *,
    assign_edges: bool = False,
    assign_nodes: bool = False,
    **kwargs
) -> None:
    """Set output edges/nodes based on inputs (take parent_output)"""
    if not (assign_edges^assign_nodes):
        raise TypeFunctionError("assign_output_axes_from_input: may not assign {assign_edges=} and {assign_nodes=}")

    inputs = tuple(node.inputs.iter(inputkey))
    outputs = tuple(node.outputs.iter(outputkey))

    for output in outputs:
        if assign_edges:
            assign_output_edges(inputs, output, **kwargs)

        if assign_nodes:
            assign_output_nodes(inputs, output, **kwargs)

def assign_outputs_axes_from_inputs(
    node: NodeT,
    inputkey: Union[str, int, slice, Sequence] = 0,
    outputkey: Union[str, int, slice, Sequence] = AllPositionals,
    *,
    assign_edges: bool = False,
    assign_nodes: bool = False,
    ignore_Nd: bool = False,
    **kwargs
) -> None:
    """Set outputs' edges/nodes based on inputs (take parent_output). Process each pair."""
    if not (assign_edges^assign_nodes):
        raise TypeFunctionError("assign_output_axes_from_input: may not assign {assign_edges=} and {assign_nodes=}")

    inputs = tuple(node.inputs.iter(inputkey))
    outputs = tuple(node.outputs.iter(outputkey))

    if len(inputs) == 1:
        inputs = repeat(inputs[0], len(outputs))

    for input, output in zip(inputs, outputs):
        if ignore_Nd and len(output.dd.shape)!=1:
            continue
        if assign_edges:
            assign_output_edges(input, output, **kwargs)

        if assign_nodes:
            assign_output_nodes(input, output, **kwargs)

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
        if not isinstance(edges, tuple):
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
        if not isinstance(edges, tuple):
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
                    output=output,
                )


def check_array_edges_consistency(
    node: NodeT, outputkey: Union[str, int, slice, Sequence] = AllPositionals
):
    """
    Checks the dimension equivalence of edges and the output, then checks that
    `len(output) = N` and `len(edges) = N+1` for each dimension.
    Tht type function is passed if the edges are empty.
    """
    for output in node.outputs.iter(outputkey):
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


def check_if_input_sorted(
    node: NodeT,
    inputkey: Union[str, int, slice, Sequence] = AllPositionals,
):
    """Checking if the inputs are sorted arrays"""
    is_sorted = lambda a: all(a[:-1] <= a[1:])
    for input in node.inputs.iter(inputkey):
        if not is_sorted(input.data):
            raise TypeFunctionError(
                "The `input` must be a sorted array!",
                node=node,
                input=input,
            )
