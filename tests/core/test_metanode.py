from numpy import linspace, pi, sin, cos
from pytest import raises

from dagflow.exception import CriticalError
from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib.base import Array
from dagflow.lib.trigonometry import Cos, Sin
from dagflow.metanode import MetaNode


def test_metanode_strategy_leading_node(testname):
    data1 = linspace(0, 1, 11, dtype="d") * pi
    data2 = linspace(1, 2, 11, dtype="d") * pi
    data3 = linspace(2, 3, 11, dtype="d") * pi
    data4 = linspace(3, 4, 11, dtype="d") * pi
    with Graph(close_on_exit=True) as graph:
        metanode = MetaNode()
        arr1 = Array("x1", data1)
        arr2 = Array("x2", data2)
        arr3 = Array("x3", data3)
        arr4 = Array("x4", data4)
        node = Cos("cos")
        arr1 >> node
        # adds an input and an output to MetaNode
        metanode._add_node(node, inputs_pos=True, outputs_pos=True)
        # auto creation of and import of input and output
        arr2 >> metanode()
        arr3 >> metanode()
        arr4 >> metanode(name="arr4", nodename="cos")
    assert len(metanode._nodes) == 1
    assert len(metanode.inputs) == 4
    assert len(metanode.outputs) == 4
    res = metanode._leading_node.outputs
    assert len(res) == 4
    assert (res[0].data == cos(data1)).all()
    assert (res[1].data == cos(data2)).all()
    assert (res[2].data == cos(data3)).all()
    assert (res[3].data == cos(data4)).all()
    savegraph(graph, f"output/{testname}.png")


def test_metanode_strategy_new_node(testname):
    data1 = linspace(0, 1, 11, dtype="d") * pi
    data2 = linspace(1, 2, 11, dtype="d") * pi
    data3 = linspace(2, 3, 11, dtype="d") * pi
    with Graph(close_on_exit=True) as graph:
        metanode = MetaNode(strategy="NewNode", new_node_cls=Cos)
        arr1 = Array("x1", data1)
        arr2 = Array("x2", data2)
        arr3 = Array("x3", data3)
        node = Cos("cos1")
        arr1 >> node
        metanode._add_node(
            node, inputs_pos=True, outputs_pos=True
        )  # adds an input and an output to MetaNode
        arr2 >> metanode(
            node_args={"name": "cos2"}
        )  # NOTE: creates a new Cos node and connect with arr2
        arr3 >> metanode(
            new_node_cls=Sin, node_args={"name": "sin"}
        )  # NOTE: creates a Sin node and connect with arr3
    assert len(metanode._nodes) == 3
    assert len(metanode.inputs) == 3
    assert len(metanode.outputs) == 3
    res = metanode._nodes
    assert len(res) == 3
    assert (res[0].outputs[0].data == cos(data1)).all()
    assert (res[1].outputs[0].data == cos(data2)).all()
    assert (res[2].outputs[0].data == sin(data3)).all()
    savegraph(graph, f"output/{testname}.png")


def test_metanode_exceptions():
    with raises(CriticalError):
        MetaNode(strategy="")
    with raises(CriticalError):
        MetaNode(strategy="LeadingNode")()
    with raises(CriticalError):
        MetaNode()(nodename="cos")
    with raises(RuntimeError):
        mn = MetaNode()
        node = Cos("cos1")
        mn._add_node(node)
        mn._add_node(node)
    with raises(RuntimeError):
        mn = MetaNode()
        node = Cos("cos1")
        node()
        mn._add_node(node, inputs_pos=True)
        mn._import_pos_inputs(node)
