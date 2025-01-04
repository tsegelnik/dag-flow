from collections.abc import Mapping, Sequence

from pytest import mark, raises

from dagflow.core.exception import ClosedGraphError, ConnectionError, UnclosedGraphError
from dagflow.core.graph import Graph
from dagflow.core.input import Input
from dagflow.core.input_handler import MissingInputAddOne
from dagflow.core.node import Node
from dagflow.core.output import Output
from dagflow.lib.common import Dummy
from dagflow.plot.graphviz import savegraph
from multikeydict.nestedmkdict import NestedMKDict


class NodeWithMIAO(Node):
    """The node with `missing_input_handler` = `MissingInputAddOne`"""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("missing_input_handler", MissingInputAddOne)
        super().__init__(*args, **kwargs)


def check_connection(obj):
    if isinstance(obj, Input):
        return obj.connected()
    if isinstance(obj, Node):
        return all(res.connected() for res in obj.inputs)
    if isinstance(obj, Sequence):
        return all(check_connection(res) for res in obj)


@mark.parametrize(
    "rhs",
    (
        Input("input", None),
        NodeWithMIAO("node"),
        [Input(f"input_{i}", None) for i in range(3)],
        [NodeWithMIAO(f"node_{i}") for i in range(3)],
        {f"i{i}": Input(f"input_{i}", None) for i in range(3)},
        {f"n{i}": NodeWithMIAO(f"node_{i}") for i in range(3)},
        {f"i{i}": [Input(f"input_{j}{i}", None) for j in range(3)] for i in range(3)},
        {f"n{i}": [NodeWithMIAO(f"node_{j}{i}") for j in range(3)] for i in range(3)},
        NestedMKDict(dic={f"i{i}": Input(f"input_{i}", None) for i in range(3)}),
        NestedMKDict(dic={f"n{i}": NodeWithMIAO(f"node_{i}") for i in range(3)}),
        NestedMKDict(
            dic={f"i{i}": [Input(f"input_{j}{i}", None) for j in range(3)] for i in range(3)}
        ),
        NestedMKDict(
            dic={f"n{i}": [NodeWithMIAO(f"node_{j}{i}") for j in range(3)] for i in range(3)}
        ),
    ),
)
def test_output_lhs(rhs):
    """
    Test a connection in the following cases:
      * `Output >> Input`;
      * `Output >> Node`;
      * `Output >> Sequence[Input]`;
      * `Output >> Sequence[Node]`;
      * `Output >> NestedMkDict[Input | Sequence[Input]]`;
      * `Output >> NestedMkDict[Node | Sequence[Node]]`;
      * `Output >> Mapping[Input | Sequence[Input]]`;
      * `Output >> Mapping[Node | Sequence[Node]]`.
    """
    lhs = Output("output", None)
    lhs >> rhs

    assert lhs.connected()
    if isinstance(rhs, (Input, Node)):
        assert check_connection(rhs)
    elif isinstance(rhs, Sequence):
        for obj in rhs:
            assert check_connection(obj)
    elif isinstance(rhs, NestedMKDict):
        for obj in rhs.walkvalues():
            assert check_connection(obj)
    elif isinstance(rhs, Mapping):
        for obj in NestedMKDict(dic=rhs).walkvalues():
            assert check_connection(obj)


def test_02():
    n1 = Node("node1")
    n2 = Node("node2")

    n1._add_output("o1")
    n1._add_output("o2")

    n2._add_input("i1")
    n2._add_input("i2")
    n2._add_output("o1")

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
