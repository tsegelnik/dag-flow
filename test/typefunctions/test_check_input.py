#!/usr/bin/env python
from dagflow.exception import TypeFunctionError
from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.input_extra import MissingInputAddOne
from dagflow.lib.Array import Array
from dagflow.lib.Dummy import Dummy
from dagflow.typefunctions import (
    AllPositionals,
    check_input_dimension,
    check_input_dtype,
    check_input_shape,
    check_input_square,
    check_input_square_or_diag,
    check_input_subtype,
    check_inputs_equivalence,
    check_inputs_multiplicable_mat,
    check_inputs_same_dtype,
    check_inputs_same_shape,
    check_output_subtype,
    copy_from_input_to_output,
)
from numpy import array, floating, integer, linspace, newaxis
from pytest import mark, raises


@mark.parametrize(
    "data,dim,shape,dtype",
    (
        ([1, 2, 3], 1, (3,), "i"),
        ([[1, 2], [3, 4]], 2, (2, 2), "d"),
        ([[[1], [2]], [[3], [4]], [[5], [6]]], 3, (3, 2, 1), "float64"),
    ),
)
def test_check_input_common(testname, debug_graph, data, dim, shape, dtype):
    with Graph(close=True, debug=debug_graph) as graph:
        arr1 = Array("arr1", array(data, dtype=dtype))
        node = Dummy(
            "node",
            missing_input_handler=MissingInputAddOne(output_fmt="result"),
        )
        arr1 >> node
        copy_from_input_to_output(node, 0, "result")
    check_input_dimension(node, 0, dim)
    check_input_shape(node, 0, shape)
    check_input_dtype(node, 0, dtype)
    with raises(TypeFunctionError):
        check_input_dimension(node, 0, dim + 1)
    with raises(TypeFunctionError):
        check_input_shape(node, 0, (1,))
    with raises(TypeFunctionError):
        check_input_dtype(node, 0, object)
    savegraph(graph, f"output/{testname}.png")


@mark.parametrize(
    "data", ([0, 1, 2], [1], [[1, 2], [1, 2, 3]], [[[], [], []]])
)
def test_check_input_square_00(testname, debug_graph, data):
    with Graph(close=False, debug=debug_graph) as graph:
        arr1 = Array("arr1", array(data, dtype=object))
        arr2 = Array("arr2", array(data, dtype=object))
        node = Dummy(
            "node",
            missing_input_handler=MissingInputAddOne(output_fmt="result"),
        )
        arr1 >> node
        arr2 >> node
    with raises(TypeFunctionError):
        check_input_square(node, 0)
    if arr1.outputs["array"].dd.dim != 1:
        with raises(TypeFunctionError):
            check_input_square_or_diag(node, 0)
    savegraph(graph, f"output/{testname}.png")


@mark.parametrize(
    "data",
    (
        linspace(0, 4, 4).reshape(2, 2),
        linspace(0, 9, 9).reshape(3, 3),
        linspace(0, 16, 16).reshape(4, 4),
    ),
)
def test_check_input_square_01(testname, debug_graph, data):
    with Graph(close=False, debug=debug_graph) as graph:
        arr1 = Array("arr1", array(data))
        node = Dummy(
            "node",
            missing_input_handler=MissingInputAddOne(output_fmt="result"),
        )
        arr1 >> node
        check_input_square(node, 0)
        check_input_square_or_diag(node, 0)
    savegraph(graph, f"output/{testname}.png")


@mark.parametrize(
    "dtype,wrongarr",
    (
        ("i", array([5, 6, 7], dtype="i")),
        ("d", array([5, 6], dtype="i")),
        ("float64", array([5, 6, 7], dtype="i")),
    ),
)
def test_check_inputs_equivalence(testname, debug_graph, dtype, wrongarr):
    # TODO: check edges and nodes
    with Graph(close=False, debug=debug_graph) as graph:
        arr1 = Array("arr1", array([1, 2], dtype=dtype))
        arr2 = Array("arr2", array([3, 4], dtype=dtype))
        arr3 = Array("arr2", array([5, 6], dtype=dtype))
        node = Dummy(
            "node",
            missing_input_handler=MissingInputAddOne(output_fmt="result"),
        )
        (arr1, arr2, arr3) >> node
        check_inputs_equivalence(node)
        check_input_shape(node, AllPositionals, (2,))
        check_input_dtype(node, AllPositionals, dtype)
        check_inputs_same_dtype(node)
        check_inputs_same_shape(node)
        Array("wrong_array", wrongarr) >> node
        with raises(TypeFunctionError):
            check_inputs_equivalence(node)
        with raises(TypeFunctionError):
            # NOTE: at least one raises Exception, see `wrongarr`
            check_inputs_same_dtype(node)
            check_inputs_same_shape(node)
    savegraph(graph, f"output/{testname}.png")


@mark.parametrize(
    "dtype",
    ("float64", "float32", "float16", "float", "double"),
)
def test_check_subtype(testname, debug_graph, dtype):
    with Graph(close=False, debug=debug_graph) as graph:
        arr1 = Array("arr1", array([1, 2], dtype=dtype))
        node = Dummy(
            "node",
            missing_input_handler=MissingInputAddOne(output_fmt="result"),
        )
        arr1 >> node
        check_input_subtype(node, 0, floating)
        check_output_subtype(node, "result", floating)
        with raises(TypeFunctionError):
            check_input_subtype(node, 0, integer)
        with raises(TypeFunctionError):
            check_output_subtype(node, "result", integer)
    savegraph(graph, f"output/{testname}.png")


@mark.parametrize(
    "data1, data2",
    (
        (linspace(0, 4, 4).reshape(2, 2), linspace(0, 4, 4)),
        (linspace(0, 9, 9), linspace(0, 3, 3)[newaxis]),
        (linspace(0, 3, 3)[newaxis].T, linspace(0, 3, 3)[newaxis].T),
    ),
)
def test_check_inputs_multiplicable_mat_00(
    testname, debug_graph, data1, data2
):
    with Graph(close=False, debug=debug_graph) as graph:
        arr1 = Array("arr1", array(data1))
        arr2 = Array("arr2", array(data2))
        node = Dummy(
            "node",
            missing_input_handler=MissingInputAddOne(output_fmt="result"),
        )
        (arr1, arr2) >> node
        with raises(TypeFunctionError):
            check_inputs_multiplicable_mat(node, 0, 1)
    savegraph(graph, f"output/{testname}.png")


@mark.parametrize(
    "data1, data2",
    (
        (linspace(0, 4, 4).reshape(2, 2), linspace(0, 4, 4).reshape(2, 2)),
        (linspace(0, 9, 9).reshape(3, 3), linspace(0, 3, 3)[newaxis].T),
        (linspace(0, 3, 3)[newaxis], linspace(0, 3, 3)[newaxis].T),
    ),
)
def test_check_inputs_multiplicable_mat_01(
    testname, debug_graph, data1, data2
):
    with Graph(close=False, debug=debug_graph) as graph:
        arr1 = Array("arr1", array(data1))
        arr2 = Array("arr2", array(data2))
        node = Dummy(
            "node",
            missing_input_handler=MissingInputAddOne(output_fmt="result"),
        )
        (arr1, arr2) >> node
        check_inputs_multiplicable_mat(node, 0, 1)
    savegraph(graph, f"output/{testname}.png")
