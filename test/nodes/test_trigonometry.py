#!/usr/bin/env python

from dagflow.graph import Graph
from dagflow.lib.Array import Array
from dagflow.lib.trigonometry import Cos, Sin, Tan, ArcCos, ArcSin, ArcTan
from dagflow.graphviz import savegraph

from numpy import allclose, arange, cos, sin, tan, arccos, arcsin, arctan
from pytest import mark


@mark.parametrize("dtype", ("d", "f"))
def test_Cos_01(testname, debug_graph, dtype):
    arrays_in = tuple(arange(12, dtype=dtype) * i for i in (1, 2, 3))

    with Graph(close=True, debug=debug_graph) as graph:
        arrays = tuple(
            Array(f"arr_{i}", array_in) for i, array_in in enumerate(arrays_in)
        )
        ccos = Cos("cos")
        arrays >> ccos

    outputs = ccos.outputs
    ress = cos(arrays_in)

    assert ccos.tainted == True
    assert all(output.dd.dtype == dtype for output in outputs)
    assert allclose(tuple(outputs.iter_data()), ress)
    assert ccos.tainted == False

    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("dtype", ("d", "f"))
def test_Sin_01(testname, debug_graph, dtype):
    arrays_in = tuple(arange(12, dtype=dtype) * i for i in (1, 2, 3))

    with Graph(close=True, debug=debug_graph) as graph:
        arrays = tuple(
            Array(f"arr_{i}", array_in) for i, array_in in enumerate(arrays_in)
        )
        ssin = Sin("sin")
        arrays >> ssin

    outputs = ssin.outputs
    ress = sin(arrays_in)

    assert ssin.tainted == True
    assert all(output.dd.dtype == dtype for output in outputs)
    assert allclose(tuple(outputs.iter_data()), ress)
    assert ssin.tainted == False

    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("dtype", ("d", "f"))
def test_Tan_01(testname, debug_graph, dtype):
    arrays_in = tuple(arange(12, dtype=dtype) * i for i in (1, 2, 3))

    with Graph(close=True, debug=debug_graph) as graph:
        arrays = tuple(
            Array(f"arr_{i}", array_in) for i, array_in in enumerate(arrays_in)
        )
        ttan = Tan("tan")
        arrays >> ttan

    outputs = ttan.outputs
    ress = tan(arrays_in)

    assert ttan.tainted == True
    assert all(output.dd.dtype == dtype for output in outputs)
    assert allclose(tuple(outputs.iter_data()), ress)
    assert ttan.tainted == False

    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("dtype", ("d", "f"))
def test_ArcCos_01(testname, debug_graph, dtype):
    arrays_in = tuple(arange(12, dtype=dtype) / 12 / i for i in (1, 2, 3))

    with Graph(close=True, debug=debug_graph) as graph:
        arrays = tuple(
            Array(f"arr_{i}", array_in) for i, array_in in enumerate(arrays_in)
        )
        aarccos = ArcCos("arccos")
        arrays >> aarccos

    outputs = aarccos.outputs
    ress = arccos(arrays_in)

    assert aarccos.tainted == True
    assert all(output.dd.dtype == dtype for output in outputs)
    assert allclose(tuple(outputs.iter_data()), ress)
    assert aarccos.tainted == False

    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("dtype", ("d", "f"))
def test_ArcSin_01(testname, debug_graph, dtype):
    arrays_in = tuple(arange(12, dtype=dtype) / 12 / i for i in (1, 2, 3))

    with Graph(close=True, debug=debug_graph) as graph:
        arrays = tuple(
            Array(f"arr_{i}", array_in) for i, array_in in enumerate(arrays_in)
        )
        aarcsin = ArcSin("sin")
        arrays >> aarcsin

    outputs = aarcsin.outputs
    ress = arcsin(arrays_in)

    assert aarcsin.tainted == True
    assert all(output.dd.dtype == dtype for output in outputs)
    assert allclose(tuple(outputs.iter_data()), ress)
    assert aarcsin.tainted == False

    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("dtype", ("d", "f"))
def test_ArcTan_01(testname, debug_graph, dtype):
    arrays_in = tuple(arange(12, dtype=dtype) * i for i in (1, 2, 3))

    with Graph(close=True, debug=debug_graph) as graph:
        arrays = tuple(
            Array(f"arr_{i}", array_in) for i, array_in in enumerate(arrays_in)
        )
        aarctan = ArcTan("arctan")
        arrays >> aarctan

    outputs = aarctan.outputs
    ress = arctan(arrays_in)

    assert aarctan.tainted == True
    assert all(output.dd.dtype == dtype for output in outputs)
    assert allclose(tuple(outputs.iter_data()), ress)
    assert aarctan.tainted == False

    savegraph(graph, f"output/{testname}.png")
