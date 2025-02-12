from collections.abc import Sequence

from pytest import mark, raises

from dagflow.core.exception import ClosedGraphError, ConnectionError, UnclosedGraphError
from dagflow.core.graph import Graph
from dagflow.core.input import Input, Inputs
from dagflow.core.node import Node
from dagflow.core.output import Output
from dagflow.lib.abstract import BlockToOneNode, OneToOneNode
from dagflow.lib.common import Dummy
from dagflow.plot.graphviz import savegraph
from multikeydict.nestedmkdict import NestedMKDict


def check_connection(obj):
    """Useful method for a check whether the object is connected or not"""
    if isinstance(obj, Input):
        return obj.connected()
    if isinstance(obj, Node):
        return all(res.connected() for res in obj.inputs)
    if isinstance(obj, (Sequence, Inputs)):
        return all(check_connection(res) for res in obj)
    return False


@mark.parametrize(
    "rhs",
    (
        Input("input", None),
        OneToOneNode("node"),
        Inputs([Input(f"input_{i}", None) for i in range(3)]),
        [Input(f"input_{i}", None) for i in range(3)],
        [OneToOneNode(f"node_{i}") for i in range(3)],
    ),
)
def test_Output_to_Input_or_Node_or_Sequence(rhs):
    """
    Test of a connection in the following cases:
      * `Output >> Input`;
      * `Output >> Node`;
      * `Output >> Inputs`;
      * `Output >> Sequence[Input]`;
      * `Output >> Sequence[Node]`;
    """
    lhs = Output("output", None)
    lhs >> rhs

    assert lhs.connected()
    if isinstance(rhs, Input):
        assert check_connection(rhs)
        assert rhs.parent_output == lhs
    elif isinstance(rhs, Node):
        assert check_connection(rhs)
        assert len(rhs.outputs) == 1
    elif isinstance(rhs, (Sequence, Inputs)):
        for obj in rhs:
            assert check_connection(obj)
            if isinstance(rhs, Input):
                assert rhs.parent_output == lhs
            elif isinstance(rhs, Node):
                assert len(obj.outputs) == 1


@mark.parametrize(
    "rhs",
    (
        {f"i{i}": Input(f"input_{i}", None) for i in range(3)},
        {f"n{i}": OneToOneNode(f"node_{i}") for i in range(3)},
        {f"i{i}": Inputs([Input(f"input_{j}{i}", None) for j in range(3)]) for i in range(3)},
        {f"i{i}": [Input(f"input_{j}{i}", None) for j in range(3)] for i in range(3)},
        {f"n{i}": [OneToOneNode(f"node_{j}{i}") for j in range(3)] for i in range(3)},
    ),
)
def test_Output_to_Mapping(rhs):
    """
    Test of a connection in the following cases:
      * `Output >> Mapping[Input | Sequence[Input] | Inputs]`;
      * `Output >> Mapping[Node | Sequence[Node]]`.
    """
    lhs = Output("output", None)
    lhs >> rhs

    assert lhs.connected()
    for obj in NestedMKDict(dic=rhs).walkvalues():
        assert check_connection(obj)


@mark.parametrize(
    "rhs",
    (
        NestedMKDict(dic={f"i{i}": Input(f"input_{i}", None) for i in range(3)}),
        NestedMKDict(dic={f"n{i}": OneToOneNode(f"node_{i}") for i in range(3)}),
        NestedMKDict(
            dic={
                f"i{i}": Inputs([Input(f"input_{j}{i}", None) for j in range(3)]) for i in range(3)
            }
        ),
        NestedMKDict(
            dic={f"i{i}": [Input(f"input_{j}{i}", None) for j in range(3)] for i in range(3)}
        ),
        NestedMKDict(
            dic={f"n{i}": [OneToOneNode(f"node_{j}{i}") for j in range(3)] for i in range(3)}
        ),
    ),
)
def test_Output_to_NestedMKDict(rhs):
    """
    Test of a connection in the following cases:
      * `Output >> NestedMkDict[Input | Sequence[Input] | Inputs]`;
      * `Output >> NestedMkDict[Node | Sequence[Node]]`;
    """
    lhs = Output("output", None)
    lhs >> rhs

    assert lhs.connected()
    for obj in rhs.walkvalues():
        assert check_connection(obj)


@mark.parametrize("lcls", (tuple, list))
@mark.parametrize("rcls", (OneToOneNode, BlockToOneNode))
def test_Sequence_Output_to_Node(lcls, rcls):
    """
    Test of a connection in the following cases:
      * `Sequence[Output] >> Node`;
      * `Sequence[Output] >> BlockToOneNode`;
    """
    lhs = lcls(Output(f"output_{i}", None) for i in range(3))
    rhs = rcls("node")
    lhs >> rhs

    assert all(obj.connected() for obj in lhs)
    assert check_connection(rhs)
    assert len(rhs.outputs) == (1 if isinstance(rhs, BlockToOneNode) else 3)


def test_NodeWith1Output_to_Node():
    """Such connections is the same as `Node.outputs[0] >> ...`"""
    n1 = Node("node1")
    n2 = Node("node2")

    n1._add_output("o1")
    n2._add_input("i1")

    n1 >> n2
    assert n2.inputs[0].connected()
    assert n1.outputs.len_all() == 1
    assert n2.inputs.len_all() == 1


@mark.parametrize("nodes_count", (1, 2, 3, 4))
def test_BlockOfNodesWith1Outputs_to_BlockToOneNode(nodes_count):
    """Such connections is the same as `(Node.outputs[0], ..., Node.outputs[0]) >> BlockToOneNode`"""
    blocknode = BlockToOneNode("BlockToOneNode")

    nodes = [Node(f"node_{i}") for i in range(nodes_count)]
    for node in nodes:
        node._add_output("o1")
    nodes >> blocknode

    assert blocknode.outputs.len_all() == 1
    assert blocknode.inputs.len_all() == nodes_count
    assert all(inp.connected() for inp in blocknode.inputs)


def test_NodeWith2Outputs_to_Node():
    """Such connections are ambiguous, so not allowed"""
    n1 = Node("node1")
    n2 = Node("node2")

    n1._add_output("o1")
    n1._add_output("o2")

    n2._add_input("i1")
    n2._add_input("i2")
    n2._add_output("o1")

    with raises(ConnectionError):
        n1 >> n2


def test_03():
    n1 = Node("node1")
    n2 = Node("node2")

    out = n1._add_output("o1")
    n2._add_input("i1")
    n2._add_output("o1")

    out >> n2


def test_04():
    n1 = Node("node1")
    n2 = Node("node2")

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
    with Graph(close_on_exit=True) as g:
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
