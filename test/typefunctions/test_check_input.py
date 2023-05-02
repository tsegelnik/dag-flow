#!/usr/bin/env python
from dagflow.exception import TypeFunctionError
from dagflow.graph import Graph
from dagflow.input_extra import MissingInputAddOne
from dagflow.lib.Array import Array
from dagflow.lib.Dummy import Dummy
from dagflow.typefunctions import (
    check_input_dimension,
    check_input_dtype,
    check_input_shape,
    check_input_square,
    check_input_square_or_diag,
    check_inputs_equivalence,
    copy_from_input_to_output,
)
from numpy import array, linspace
from pytest import mark, raises


@mark.parametrize(
    "data,dim,shape,dtype",
    (
        ([1, 2, 3], 1, (3,), "i"),
        ([[1, 2], [3, 4]], 2, (2, 2), "d"),
        ([[[1], [2]], [[3], [4]], [[5], [6]]], 3, (3, 2, 1), "float64"),
    ),
)
def test_check_input_common(debug_graph, data, dim, shape, dtype):
    with Graph(close=True, debug=debug_graph):
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


@mark.parametrize(
    "data", ([0, 1, 2], [1], [[1, 2], [1, 2, 3]], [[[], [], []]])
)
def test_check_input_square_00(debug_graph, data):
    with Graph(close=False, debug=debug_graph):
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
        check_input_square_or_diag(node, 0)


@mark.parametrize(
    "data",
    (
        linspace(0, 4, 4).reshape(2, 2),
        linspace(0, 9, 9).reshape(3, 3),
        linspace(0, 16, 16).reshape(4, 4),
    ),
)
def test_check_input_square_01(debug_graph, data):
    with Graph(close=False, debug=debug_graph):
        arr1 = Array("arr1", array(data))
        node = Dummy(
            "node",
            missing_input_handler=MissingInputAddOne(output_fmt="result"),
        )
        arr1 >> node
        check_input_square(node, 0)
        check_input_square_or_diag(node, 0)


@mark.parametrize("dtype,wrongarr", (("i",array([5, 6, 7], dtype="i")), ("d", array([5, 6], dtype="i")), ("float64", array([5, 6, 7], dtype="i"))))
def test_check_inputs_equivalence(debug_graph, dtype,wrongarr):
    #TODO: check edges and nodes
    with Graph(close=False, debug=debug_graph):
        arr1 = Array("arr1", array([1, 2], dtype=dtype))
        arr2 = Array("arr2", array([3, 4], dtype=dtype))
        arr3 = Array("arr2", array([5, 6], dtype=dtype))
        node = Dummy(
            "node",
            missing_input_handler=MissingInputAddOne(output_fmt="result"),
        )
        (arr1, arr2, arr3) >> node
        check_inputs_equivalence(node)
        arr4 = Array("arr4", wrongarr)
        arr4 >> node
        with raises(TypeFunctionError):
            check_inputs_equivalence(node)
