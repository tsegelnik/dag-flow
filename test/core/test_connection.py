#!/usr/bin/env python
from pytest import raises

from dagflow.exception import ClosedGraphError
from dagflow.exception import ConnectionError
from dagflow.exception import UnclosedGraphError
from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.input import Input
from dagflow.lib.Dummy import Dummy
from dagflow.nodes import FunctionNode
from dagflow.output import Output


def test_01():
    i = Input("input", None)
    o = Output("output", None)

    o >> i


def test_02():
    n1 = FunctionNode("node1")
    n2 = FunctionNode("node2")

    n1._add_output("o1")
    n1._add_output("o2")

    n2._add_input("i1")
    n2._add_input("i2")
    n2._add_output("o1")

    n1 >> n2


def test_03():
    n1 = FunctionNode("node1")
    n2 = FunctionNode("node2")

    out = n1._add_output("o1")

    n2._add_input("i1")
    n2._add_output("o1")

    out >> n2


def test_04():
    n1 = FunctionNode("node1")
    n2 = FunctionNode("node2")

    out = n1._add_output("o1")

    n2._add_pair("i1", "o1")

    out >> n2


def test_05():
    n1 = Dummy("node1")
    n2 = Dummy("node2")

    out1 = n1._add_output("o1", allocatable=False)
    out2 = n1._add_output("o2", allocatable=False)

    _, final = n2._add_pair("i1", "o1", output_kws={"allocatable": False})
    n2._add_input("i2")

    (out1, out2) >> n2

    n2.close()
    assert n2.closed
    assert n1.closed
    with raises(ClosedGraphError):
        n2.add_input("i3")
    with raises(ClosedGraphError):
        n1.add_output("o3")
    final.data


def test_06():
    n1 = Dummy("node1")
    n2 = Dummy("node2")

    out1 = n1._add_output("o1", allocatable=False)
    out2 = n1._add_output("o2", allocatable=False)

    _, final = n2._add_pair("i1", "o1", output_kws={"allocatable": False})
    n2._add_input("i2")

    (out1, out2) >> n2

    n1.close(recursive=False)
    assert n1.closed
    assert not n2.closed
    n2.close(recursive=False)
    assert n2.closed
    with raises(ClosedGraphError):
        n2.add_input("i3")
    with raises(ClosedGraphError):
        n1.add_output("o3")
    final.data


def test_07():
    with Graph() as g:
        n1 = Dummy("node1")
        n2 = Dummy("node2")

    out1 = n1._add_output("o1", allocatable=False)
    out2 = n1._add_output("o2", allocatable=False)

    _, final = n2._add_pair("i1", "o1", output_kws={"allocatable": False})
    n2._add_input("i2")

    (out1, out2) >> n2

    with raises(UnclosedGraphError):
        final.data
    g.close()
    with raises(ClosedGraphError):
        n2.add_input("i3")
    with raises(ClosedGraphError):
        n1.add_output("o3")
    final.data


def test_08():
    with Graph() as g:
        n1 = Dummy("node1")
        n2 = Dummy("node2")
        n3 = Dummy("node3")

    out1 = n1._add_output("o1", allocatable=False)
    out2 = n1._add_output("o2", allocatable=False)

    _, out3 = n2._add_pair("i1", "o1", output_kws={"allocatable": False})
    n2._add_input("i2")

    _, final = n3._add_pair("i1", "o1", output_kws={"allocatable": False})

    (out1, out2) >> n2
    out3 >> n3

    with raises(UnclosedGraphError):
        final.data
    g.close()
    with raises(ClosedGraphError):
        n2.add_input("i3")
    with raises(ClosedGraphError):
        n1.add_output("o3")
    with raises(ClosedGraphError):
        n3.add_pair("i3", "o3")
    final.data

    print()
    final.data

    print("Taint n2")
    n2.taint()
    final.data

    print("Taint n3")
    n3.taint()
    final.data


def test_09(testname):
    """Test <<"""
    with Graph(close=True) as g:
        n1 = Dummy("node1")
        n2 = Dummy("node2")

        out11 = n1._add_output("o1", allocatable=False)
        out12 = n1._add_output("o2", allocatable=False)
        out13 = n1._add_output("o3", allocatable=False)

        n2._add_input("i1")
        in22 = n2._add_input("i2")
        n2._add_input("i3")
        out2 = n2._add_output("o2", allocatable=False)

        out12 >> in22

        with raises(ConnectionError):
            n2 << {"i1": object()}

        with raises(ConnectionError):
            n2 << {"i2": object()}

        n2 << {
            "i1": out11,
            "i2": out12,
            "i3": out13,
        }

    out2.data

    savegraph(g, f"output/{testname}.pdf")
