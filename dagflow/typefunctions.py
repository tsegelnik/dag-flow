from __future__ import annotations

from collections.abc import Sequence
from itertools import repeat
from typing import TYPE_CHECKING

from numpy import allclose, issubdtype, result_type

from .exception import TypeFunctionError
from .input import Input
from .output import Output

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import DTypeLike

    from .node import Node

AllPositionals = slice(None)
LimbKey = str | int | slice | Sequence

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


def cpy_meshes(input, output):
    output.dd.axes_meshes = input.dd.axes_meshes


def check_has_inputs(
    node: Node,
    inputkey: str | int | slice | Sequence | None = None,
    check_named: bool = False,
) -> None:
    """Checking if the node has inputs"""
    if inputkey is None or inputkey == AllPositionals:
        try:
            node.inputs[0]
        except Exception as exc:
            if not check_named:
                raise TypeFunctionError(
                    "The node must have at least one positional input!", node=node
                ) from exc
            if node.inputs.len_kw() == 0:
                raise TypeFunctionError(
                    "The node must have at least one input!", node=node
                ) from exc
    else:
        try:
            node.inputs[inputkey]
        except Exception as exc:
            raise TypeFunctionError(
                f"The node must have the input '{inputkey}'!", node=node
            ) from exc


def check_inputs_number(node: Node, n: int) -> None:
    """Checking if the node has only `n` inputs"""
    if (ninp := len(node.inputs)) != n:
        raise TypeFunctionError(f"The node must have only {n} inputs, but given {ninp}!", node=node)


def check_outputs_number(node: Node, n: int) -> None:
    """Checking if the node has only `n` outputs"""
    if (nout := len(node.outputs)) != n:
        raise TypeFunctionError(
            f"The node must have only {n} outputs, but given {nout}!", node=node
        )


def copy_from_input_to_output(
    node: Node,
    inputkey: LimbKey = 0,
    outputkey: LimbKey = AllPositionals,
    *,
    dtype: bool = True,
    shape: bool = True,
    edges: bool = True,
    meshes: bool = True,
    prefer_largest_input: bool = False,
    prefer_input_with_edges: bool = False,
) -> None:
    """Coping input dtype and setting for the output"""
    if not any((dtype, shape, edges, meshes)):
        return

    inputs = tuple(node.inputs.iter(inputkey))
    outputs = tuple(node.outputs.iter(outputkey))

    caller = MethodSequenceCaller()
    if dtype:
        caller.add(cpy_dtype)
    if shape:
        caller.add(cpy_shape)
    if edges:
        caller.add(cpy_edges)
    if meshes:
        caller.add(cpy_meshes)

    has_preference = prefer_input_with_edges or prefer_largest_input
    if has_preference and len(inputs) > 1:
        largest_input = inputs[0]
        largest_size = largest_input.dd.size
        found_edges = bool(largest_input.dd.axes_edges)
        for input in inputs[1:]:
            if (newsize := input.dd.size) <= largest_size and prefer_largest_input:
                continue
            if prefer_input_with_edges and found_edges and not bool(input.dd.axes_edges):
                continue
            largest_size = newsize
            largest_input = input
            found_edges = bool(input.dd.axes_edges)
        inputs = (largest_input,)

    if len(inputs) == 1:
        inputs = repeat(inputs[0], len(outputs))

    for input, output in zip(inputs, outputs, strict=True):
        caller(input, output)


def copy_input_dtype_to_output(
    node: Node,
    inputkey: LimbKey = 0,
    outputkey: LimbKey = AllPositionals,
) -> None:
    """Coping input dtype and setting for the output"""
    inputs = tuple(node.inputs.iter(inputkey))
    outputs = tuple(node.outputs.iter(outputkey))

    if len(inputs) == 1:
        inputs = repeat(inputs[0], len(outputs))

    for input, output in zip(inputs, outputs, strict=True):
        output.dd.dtype = input.dd.dtype


def eval_output_dtype(
    node: Node,
    inputkey: LimbKey = AllPositionals,
    outputkey: LimbKey = AllPositionals,
) -> None:
    """Automatic calculation and setting dtype for the output"""
    inputs = node.inputs.iter(inputkey)
    outputs = node.outputs.iter(outputkey)

    dtype = result_type(*(inp.dd.dtype for inp in inputs))
    for output in outputs:
        output.dd.dtype = dtype


def copy_input_shape_to_outputs(
    node: Node,
    inputkey: str | int = 0,
    outputkey: LimbKey = AllPositionals,
) -> None:
    """Coping input shape and setting to each output"""
    inputs = tuple(node.inputs.iter(inputkey))
    outputs = tuple(node.outputs.iter(outputkey))

    if len(inputs) == 1:
        inputs = repeat(inputs[0], len(outputs))

    for input, output in zip(inputs, outputs, strict=True):
        output.dd.shape = input.dd.shape


def check_input_dimension(node: Node, inputkey: LimbKey, ndim: int, **kwargs):
    """Checking the dimension of the input"""
    for input in node.inputs.iter(inputkey, **kwargs):
        dim = len(input.dd.shape)
        if ndim != dim:
            raise TypeFunctionError(
                f"The node supports only {ndim}d inputs. Got {dim}d!",
                node=node,
                input=input,
            )


def check_input_shape(node: Node, inputkey: LimbKey, shape: tuple, **kwargs):
    """Checking the shape equivalence for inputs"""
    for input in node.inputs.iter(inputkey, **kwargs):
        sshape = input.dd.shape
        if sshape != shape:
            raise TypeFunctionError(
                f"The node supports only inputs with shape={shape}. Got {sshape}!",
                node=node,
                input=input,
            )


def check_input_size(
    node: Node,
    inputkey: LimbKey,
    *,
    exact: int | None = None,
    min: int | None = None,
    max: int | None = None,
    **kwargs,
):
    """Checking the shape equivalence for inputs"""
    if exact is not None:
        for input in node.inputs.iter(inputkey, **kwargs):
            size = input.dd.size
            if size != exact:
                raise TypeFunctionError(
                    f"The input size {size} is not equal to {exact}",
                    node=node,
                    input=input,
                )
    else:
        if min is None and max is None:
            raise TypeFunctionError(
                "`exact`, `min` and/or `max` must be passed into the type function!",
                node=node,
            )
        for input in node.inputs.iter(inputkey, **kwargs):
            size = input.dd.size
            if min is not None and size < min:
                raise TypeFunctionError(
                    f"The input size {size} is below {min}",
                    node=node,
                    input=input,
                )
            if max is not None and size > max:
                raise TypeFunctionError(
                    f"The input size {size} is above {max}",
                    node=node,
                    input=input,
                )


def check_input_dtype(node: Node, inputkey: LimbKey, dtype, **kwargs):
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
    node: Node,
    inputkey: LimbKey,
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


def _check_input_block_or_diag(node: Node, input: Input, *, check_square: bool = False) -> int:
    """Check if input is a block/square matrix or a diagonal (1d) of a square matrix.
    Returns the maximal dimension."""
    shape = input.dd.shape
    dim = len(shape)
    if dim > 2:
        raise TypeFunctionError(
            f"The node supports only 1d or 2d. Got {dim}d!",
            node=node,
            input=input,
        )
    if dim == 2:
        if check_square and shape[0] != shape[1]:
            raise TypeFunctionError(
                f"The node supports only square inputs (or 1d as diagonal). Got {shape}!",
                node=node,
                input=input,
            )
    elif dim != 1:
        mtype = check_square and "square" or "matrix"
        raise TypeFunctionError(
            f"The node supports only {mtype} inputs (or 1d as diagonal). Got {shape}!",
            node=node,
            input=input,
        )

    return dim


def check_input_matrix_or_diag(node: Node, inputkey: LimbKey, check_square: bool = False) -> int:
    """Check if input is a square matrix or diagonal (1d) of a square matrix.
    Returns the maximal dimension."""
    dim_max = 0
    for input in node.inputs.iter(inputkey):
        dim = _check_input_block_or_diag(node, input, check_square=check_square)
        dim_max = max(dim, dim_max)
    return dim_max


def check_inputs_consistent_square_or_diag(node: Node, inputkey: LimbKey = AllPositionals) -> int:
    """Check if inputs are square matrices or diagonals (1d) of a square matrices of the same size.
    Returns the maximal dimension."""
    inputs = tuple(node.inputs.iter(inputkey))

    dim_max = 0
    shape0 = inputs[0].dd.shape[0]

    for input in inputs:
        dim = _check_input_block_or_diag(node, input, check_square=True)
        dim_max = max(dim, dim_max)

        shape = input.dd.shape[0]
        if shape != shape0:
            raise TypeFunctionError(
                f"All inputs should have the same size {shape0}, got {shape}",
                node=node,
                input=input,
            )

    return dim_max


def shapes_are_broadcastable(shape1: Sequence[int], shape2: Sequence[int]) -> bool:
    return all(a == 1 or b == 1 or a == b for a, b in zip(shape1[::-1], shape2[::-1]))


def check_inputs_equivalence(
    node: Node,
    inputkey: LimbKey = AllPositionals,
    *,
    check_dtype: bool = True,
    check_shape: bool = True,
    check_edges: bool = True,
    check_edges_contents: bool = False,
    check_meshes: bool = False,
    check_meshes_contents: bool = False,
    broadcastable: bool = False,
    atol: float = 1.0e-14,
):
    """Checking the equivalence of the dtype, shape, axes_edges and axes_meshes of all the inputs"""
    inputs = tuple(node.inputs.iter(inputkey))
    input0, inputs = inputs[0], inputs[1:]

    dtype, shape, edges, meshes = (
        input0.dd.dtype,
        input0.dd.shape,
        input0.dd.axes_edges,
        input0.dd.axes_meshes,
    )
    for input in inputs:
        dd = input.dd
        dtype_inconsistent = check_dtype and dd.dtype != dtype
        if check_shape:
            shape_inconsistent = (
                not shapes_are_broadcastable(shape, dd.shape)
                if broadcastable
                else dd.shape != shape
            )
        else:
            shape_inconsistent = False

        edges_inconsistent = check_edges and dd.axes_edges and edges and dd.axes_edges != edges
        if edges_inconsistent and check_edges_contents:
            edges_inconsistent = not all(
                allclose(a.data, b.data, atol=atol) for a, b in zip(edges, dd.axes_edges)
            )

        meshes_inconsistent = (
            check_meshes and dd.axes_meshes and meshes and dd.axes_meshes != meshes
        )
        if meshes_inconsistent and check_meshes_contents:
            meshes_inconsistent = not all(
                allclose(a.data, b.data, atol=atol) for a, b in zip(meshes, dd.axes_meshes)
            )

        if any(
            (
                dtype_inconsistent,
                shape_inconsistent,
                edges_inconsistent,
                meshes_inconsistent,
            )
        ):
            raise TypeFunctionError(
                f"Input data [{dd.dtype=}, {dd.shape=}, {dd.axes_edges=}, {dd.axes_meshes=}]"
                f" is inconsistent with [{dtype=}, {shape=}, {edges=}, {meshes=}]",
                node=node,
                input=input,
            )


def check_inputs_same_dtype(node: Node, inputkey: LimbKey = AllPositionals) -> DTypeLike:
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
    return dtype


def check_inputs_same_shape(node: Node, inputkey: LimbKey = AllPositionals) -> tuple[int, ...]:
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
    return shape


def check_input_subtype(node: Node, inputkey: LimbKey, dtype: DTypeLike):
    """Checks if the input dtype is some subtype of `dtype`."""
    for input in node.inputs.iter(inputkey):
        if not issubdtype(input.dd.dtype, dtype):
            raise TypeFunctionError(
                f"The input must be an array of {dtype}, but given '{input.dd.dtype}'!",
                node=node,
                input=input,
            )


def check_output_subtype(node: Node, outputkey: LimbKey, dtype: DTypeLike):
    """Checks if the output dtype is some subtype of `dtype`."""
    for output in node.outputs.iter(outputkey):
        if not issubdtype(output.dd.dtype, dtype):
            raise TypeFunctionError(
                f"The output must be an array of {dtype}, but given '{output.dd.dtype}'!",
                node=node,
                output=output,
            )


def check_inputs_multiplicable_mat(
    node: Node, inputkey1: LimbKey, inputkey2: LimbKey
) -> tuple[tuple[int, int], ...]:
    """Checking that inputs from key1 and key2 may be multiplied (matrix)
    Return shapes of the multiplications.
    """
    inputs1 = tuple(node.inputs.iter(inputkey1))
    inputs2 = tuple(node.inputs.iter(inputkey2))

    len1, len2 = len(inputs1), len(inputs2)
    if len1 == len2:
        pass
    elif len1 == 1:
        inputs1 = repeat(inputs1[0], len2)
    elif len2 == 1:
        inputs2 = repeat(inputs2[0], len1)

    ret = []
    for input1, input2 in zip(inputs1, inputs2, strict=True):
        shape1 = input1.dd.shape
        shape2 = input2.dd.shape
        if shape1[-1] != shape2[0]:
            raise TypeFunctionError(
                f"Inputs {shape1} and {shape2} are not multiplicable",
                node=node,
                input=input,
            )
        ret.append((shape1[0], shape2[-1]))

    return tuple(ret)


def copy_input_edges_to_output(
    node: Node,
    inputkey: str | int = 0,
    outputkey: LimbKey = AllPositionals,
) -> None:
    """Coping input edges and setting for the output"""
    inputs = tuple(node.inputs.iter(inputkey))
    outputs = tuple(node.outputs.iter(outputkey))

    if len(inputs) == 1:
        inputs = repeat(inputs[0], len(outputs))

    for input, output in zip(inputs, outputs, strict=True):
        output.dd.axes_edges = input.dd.axes_edges


def assign_output_edges(
    input: Input | Sequence[Input], output: Output, ignore_assigned: bool = False
):
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

    if len(dd.shape) != len(edges):
        raise TypeFunctionError(
            f"Output ndim={len(dd.shape)} is inconsistent with edges ndim={len(edges)}",
            output=output,
            input=input,
        )

    for i, (dimsize, edgeoutput) in enumerate(zip(dd.shape, edges)):
        shapeedges = edgeoutput.dd.shape
        if len(shapeedges) != 1:
            raise TypeFunctionError(
                f"Edges of {i}th dimension has non-1d shape", output=output, input=input
            )
        dimedges = shapeedges[0]
        if dimsize != (dimedges - 1):
            raise TypeFunctionError(
                f"Output dimension {i} size={dimsize} is inconsistent with edges"
                f" size={len(dimedges)}",
                output=output,
                input=input,
            )

    output.dd.axes_edges = tuple(edges)


def assign_output_meshes(
    input: Input | Sequence[Input],
    output: Output,
    *,
    ignore_assigned: bool = False,
    overwrite_assigned: bool = False,
    ignore_inconsistent_number_of_meshes: bool = False,
):
    """Assign output's edges from input's parent output"""
    dd = output.dd

    if dd.axes_meshes:
        if ignore_assigned:
            return
        elif not overwrite_assigned:
            raise TypeFunctionError("Meshes already assigned", output=output, input=input)

    if isinstance(input, Input):
        meshes = [input.parent_output]
    else:
        meshes = [inp.parent_output for inp in input]

    naxes = len(dd.shape)
    if naxes != len(meshes) and (not overwrite_assigned or naxes != len(dd.axes_meshes)):
        if ignore_inconsistent_number_of_meshes:
            return
        raise TypeFunctionError(
            f"Output ndim={len(dd.shape)} is inconsistent with meshes ndim={len(meshes)}",
            output=output,
            input=input,
        )

    for i, meshesoutput in enumerate(meshes):
        if dd.shape != meshesoutput.dd.shape:
            raise TypeFunctionError(
                f"Output shape={dd.shape} is inconsistent with meshes"
                f" {i} shape={meshesoutput.dd.shape}",
                output=output,
                input=input,
            )

    if overwrite_assigned:
        newmeshes = list(output.dd.axes_meshes)
        for i, mesh in enumerate(meshes):
            try:
                newmeshes[i] = mesh
            except IndexError:
                newmeshes.append(mesh)
        dd.axes_meshes = tuple(newmeshes)
    else:
        dd.axes_meshes = tuple(meshes)


def assign_output_axes_from_inputs(
    node: Node,
    inputkey: LimbKey = 0,
    outputkey: LimbKey = AllPositionals,
    *,
    assign_edges: bool = False,
    assign_meshes: bool = False,
    **kwargs,
) -> None:
    """Set output edges/meshes based on inputs (take parent_output)"""
    if not (assign_edges ^ assign_meshes):
        raise TypeFunctionError(
            "assign_output_axes_from_input: may not assign {assign_edges=} and {assign_meshes=}"
        )

    inputs = tuple(node.inputs.iter(inputkey))
    outputs = tuple(node.outputs.iter(outputkey))

    for output in outputs:
        if assign_edges:
            assign_output_edges(inputs, output, **kwargs)

        if assign_meshes:
            assign_output_meshes(inputs, output, **kwargs)


def assign_outputs_axes_from_inputs(
    node: Node,
    inputkey: LimbKey = 0,
    outputkey: LimbKey = AllPositionals,
    *,
    assign_edges: bool = False,
    assign_meshes: bool = False,
    ignore_Nd: bool = False,
    **kwargs,
) -> None:
    """Set outputs' edges/meshes based on inputs (take parent_output). Process each pair."""
    if not (assign_edges ^ assign_meshes):
        raise TypeFunctionError(
            "assign_output_axes_from_input: may not assign {assign_edges=} and {assign_meshes=}"
        )

    inputs = tuple(node.inputs.iter(inputkey))
    outputs = tuple(node.outputs.iter(outputkey))

    if len(inputs) == 1:
        inputs = repeat(inputs[0], len(outputs))

    for input, output in zip(inputs, outputs):
        if ignore_Nd and len(output.dd.shape) != 1:
            continue
        if assign_edges:
            assign_output_edges(input, output, **kwargs)

        if assign_meshes:
            assign_output_meshes(input, output, **kwargs)


def check_input_edges_dim(
    node: Node,
    inputkey: LimbKey = AllPositionals,
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
    node: Node,
    inputkey: LimbKey = AllPositionals,
    reference: tuple[Output] | None = None,
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
    node: Node,
    inputkey: LimbKey = AllPositionals,
    outputkey: LimbKey = AllPositionals,
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


def check_array_edges_consistency(node: Node, outputkey: LimbKey = AllPositionals):
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
                        f"Array: the data length+1 ({dd.shape[i]+1}) "
                        f"is not consistent with edges ({edge.dd.shape[0]})",
                        node=node,
                        output=output,
                    )


def find_max_size_of_inputs(node: Node, inputkey: LimbKey = AllPositionals) -> int:
    """Find maximum size of inputs"""
    inputs = tuple(node.inputs.iter(inputkey))
    input0, inputs = inputs[0], inputs[1:]

    size = input0.dd.size
    for input in inputs:
        if (y := input.dd.size) > size:
            size = y
    return size


def check_inputs_multiplicity(node: Node, N: int) -> None:
    """Check whether inputs count multiple to some N or not"""
    n = node.inputs.len_pos()
    if n % N != 0:
        raise TypeFunctionError(f"Node takes only {N}N inputs, but given {n}", node=node)


# NOTE: may be it will be needed later, but for now is not
# def check_if_input_sorted(
#    node: Node,
#    inputkey: LimbKey = AllPositionals,
# ):
#    """Checking if the inputs are sorted arrays"""
#    is_sorted = lambda a: all(a[:-1] <= a[1:])
#    for input in node.inputs.iter(inputkey):
#        if not is_sorted(input.data):
#            raise TypeFunctionError(
#                "The `input` must be a sorted array!",
#                node=node,
#                input=input,
#            )
